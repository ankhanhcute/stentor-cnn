"""
predict.py
----------
Run a pretrained model on a new unlabeled dataset.
Outputs 0/1/null per stimulus + uncertain stimuli JSON.

Uncertainty is flagged if EITHER:
  - probability margin uncertainty (close to deploy_thresh), OR
  - 5-fold ensemble disagreement (the fold models don't agree on the hard call)

Each prediction also carries holdfast_method / low_confidence_input:
  find_holdfast falls back to cell_length / nearest_mask_pixel / fallback_center
  when no single-frame "ball" candidate could be resolved. This happens
  disproportionately in crowded scenes (multiple overlapping cells) where
  no static frame has enough info to identify the right structure. This is
  an INPUT-space flag, distinct from probability-margin/vote-split
  uncertainty: a stimulus can be far from threshold and fold-unanimous and
  still be unreliable because the crop itself may not contain the real cell.

Usage:
    python predict.py <tiled.h5> <meta.h5> [checkpoint.pt]

Example:
    python predict.py ../tiles/2025_01_10_tiled.h5 ../meta/2025_01_10_tiled_data.h5
"""
import sys, os, json
import torch
from torch.utils.data import DataLoader
import loader
from model import StentorSequenceModel, count_params
from evaluate import predict_all_core
from find_holdfast import find_holdfast
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

#-----Config----
DROPOUT = 0.3
CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
THRESH_FILE = os.path.join(THIS_DIR, "checkpoints", "best_thresh.json")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CROSS_VAL_DIR = os.path.join(THIS_DIR, "..", "cross_val", "checkpoints")
N_FOLDS = 5

#----main----
def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device     : {device}")
    print(f"checkpoint : {CHECKPOINT}")
    print(f"dataset    : {TILED_H5}\n")

    with open(THRESH_FILE) as f:
        best_thresh = json.load(f)["threshold"]

    model = StentorSequenceModel(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    print(f"model params: {count_params(model):,}\n")

    fold_models = []
    for fold_idx in range(N_FOLDS):
        fold_ckpt = os.path.join(CROSS_VAL_DIR, f"fold_{fold_idx}", "best_model.pt")
        fm = StentorSequenceModel(dropout=DROPOUT).to(device)
        fm.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
        fm.eval()
        fold_models.append(fm)
    print(f"fold models loaded: {len(fold_models)}\n")

    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    total_stims = meta["total_stims"]
    manual      = np.full((meta["num_cells"], total_stims), np.nan)
    all_cells   = list(range(meta["num_cells"]))
    ds = loader.StentorPairs(tiles, manual, all_cells, tiled_h5_path=TILED_H5)

    print(f"cells loaded: {len(ds)}")

    all_predictions, uncertain_stimuli = predict_all_core(model, [ds], device, best_thresh)

    pred_map = {(p["cell"], p["stimulus"]): p for p in all_predictions}
    n_low_conf_cells = 0
    with torch.no_grad():
        for i, c in enumerate(ds.index):
            seq, _ = ds[i]

            # --- holdfast confidence flag (input-space, computed once per cell) ---
            tiles_cell = tiles[c]
            crop_size = tiles_cell.shape[0]
            hf = find_holdfast(tiles_cell, crop_size)
            hf_method = hf.get("method", "?")
            low_conf_input = not hf_method.startswith("ball")
            if low_conf_input:
                n_low_conf_cells += 1

            seq_batched = seq.unsqueeze(0).to(device)
            fold_probs = []
            for fm in fold_models:
                p = torch.sigmoid(fm(seq_batched)).squeeze().cpu().numpy()
                fold_probs.append(p)
            fold_probs = np.stack(fold_probs, axis=0)
            fold_votes = (fold_probs >= best_thresh).astype(int)
            vote_sum = fold_votes.sum(axis=0)
            for k in range(total_stims):
                vs = int(vote_sum[k])
                vote_split = (vs != 0) and (vs != N_FOLDS)
                entry = pred_map[(c, k)]
                entry["vote_split"] = vote_split
                entry["fold_vote_sum"] = vs
                entry["holdfast_method"] = hf_method
                entry["low_confidence_input"] = low_conf_input
                if vote_split and entry["prediction"] is not None:
                    entry["prediction"] = None
                    uncertain_stimuli.append({
                        "cell": c,
                        "stimulus": k,
                        "probability": entry["probability"]
                    })

    n_certain   = sum(1 for p in all_predictions if p["prediction"] is not None)
    n_uncertain = sum(1 for p in all_predictions if p["prediction"] is None)

    print(f"certain     : {n_certain}")
    print(f"uncertain   : {n_uncertain}")
    print(f"low_confidence_input cells : {n_low_conf_cells} / {len(ds.index)}")

    dataset_name = os.path.basename(TILED_H5).replace("_tiled.h5", "")
    pred_path = os.path.join(OUT_DIR, f"predictions_{dataset_name}.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\npredictions saved → {pred_path}")

    unc_path = os.path.join(OUT_DIR, f"uncertain_{dataset_name}.json")
    with open(unc_path, "w") as f:
        json.dump(uncertain_stimuli, f, indent=2)
    print(f"uncertain saved   → {unc_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <tiled.h5> <meta.h5> [checkpoint.pt]")
        sys.exit(1)
    TILED_H5 = sys.argv[1]
    META_H5 = sys.argv[2]
    if len(sys.argv) > 3:
        CHECKPOINT = sys.argv[3]
    main()
