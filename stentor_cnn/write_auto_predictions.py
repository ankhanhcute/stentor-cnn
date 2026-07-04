"""
write_auto_predictions.py

Run the trained StentorSequenceModel on a dataset and write predictions
into the 'auto' dataset inside {stamp}_contractions.h5. 'manual' is never
touched if it already exists.

Output values:
- 0.0: model predicts not contracted
- 1.0: model predicts contracted

NOTE: the pre-contracted (NaN) flag has been REMOVED for now. It was found
to false-flag ~25% of stimuli on real data.

Usage:
    python write_auto_predictions.py <tiled.h5> <meta.h5> [contractions.h5] [checkpoint.pt]

    If contractions.h5 is omitted, auto-derived supporting two layouts:
      A) {fold_name}/{fold_name}_tiled.h5        -> annotated/ sibling of {fold_name}/
      B) {fold_name}/tiled/{fold_name}_tiled.h5  -> annotated/ sibling of tiled/
         (B matches cell_annotation.jl's exact convention)
"""
import sys, os
import json
import h5py
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import loader
from model import StentorSequenceModel

CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
THRESH_FILE = os.path.join(THIS_DIR, "checkpoints", "best_thresh.json")


def main():
    if len(sys.argv) < 3:
        print("Usage: python write_auto_predictions.py <tiled.h5> <meta.h5> [contractions.h5] [checkpoint.pt]")
        sys.exit(1)

    tiled_h5 = sys.argv[1]
    meta_h5 = sys.argv[2]

    if len(sys.argv) > 3:
        contractions_h5 = sys.argv[3]
    else:
        tiled_dir = os.path.dirname(os.path.abspath(tiled_h5))
        stamp = os.path.basename(tiled_h5).replace("_tiled.h5", "")
        if os.path.basename(tiled_dir) == "tiled":
            fold_dir = os.path.dirname(tiled_dir)
        else:
            fold_dir = tiled_dir
        annotated_dir = os.path.join(fold_dir, "annotated")
        contractions_h5 = os.path.join(annotated_dir, f"{stamp}_contractions.h5")
        print("no contractions.h5 given, auto-derived path:")
        print(f"  {contractions_h5}")

    checkpoint = sys.argv[4] if len(sys.argv) > 4 else CHECKPOINT

    file_existed = os.path.exists(contractions_h5)
    manual_shape = None
    had_auto = False
    old_auto_shape = None

    if file_existed:
        with h5py.File(contractions_h5, "r") as f:
            had_auto = "auto" in f
            old_auto_shape = f["auto"].shape if had_auto else None
            if "manual" in f:
                manual_shape = f["manual"].shape

    print(f"contractions file : {contractions_h5}")
    if file_existed:
        print(f"  existing manual shape : {manual_shape if manual_shape else '(none)'}")
        print(f"  existing auto shape   : {old_auto_shape if had_auto else '(none)'}")
    else:
        print("  file does not exist yet, will be created with 'auto' only")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device     : {device}")
    print(f"checkpoint : {checkpoint}")

    with open(THRESH_FILE) as f:
        best_thresh = json.load(f)["threshold"]
    print(f"threshold  : {best_thresh}")

    model = StentorSequenceModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    tiles, meta = loader.load_tiles(tiled_h5, meta_h5)
    num_cells = meta["num_cells"]
    total_stims = meta["total_stims"]

    dummy_manual = np.full((num_cells, total_stims), np.nan)
    all_cells = list(range(num_cells))
    ds = loader.StentorPairs(tiles, dummy_manual, all_cells, tiled_h5_path=tiled_h5)

    if manual_shape is not None:
        expected_total_stim = manual_shape[1] if manual_shape[0] == num_cells else manual_shape[0]
        if expected_total_stim != total_stims:
            print(f"WARNING: manual implies total_stim={expected_total_stim}, "
                  f"tiles imply {total_stims}. Using tiles-derived value.")

    predictions = np.zeros((num_cells, total_stims), dtype=np.float64)

    print(f"running inference on {len(ds)} cells")
    with torch.no_grad():
        for i, c in enumerate(ds.index):
            seq, _ = ds[i]
            seq_batched = seq.unsqueeze(0).to(device)
            probs = torch.sigmoid(model(seq_batched)).squeeze().cpu().numpy()
            preds = (probs >= best_thresh).astype(np.float64)
            predictions[c, :] = preds

    print(f"predictions shape (Python, num_cells x total_stim): {predictions.shape}")
    print(f"  will read back in Julia as {predictions.shape[::-1]}, matching manual's convention")

    os.makedirs(os.path.dirname(os.path.abspath(contractions_h5)), exist_ok=True)
    mode = "r+" if file_existed else "w"
    with h5py.File(contractions_h5, mode) as f:
        if "auto" in f:
            del f["auto"]
        f.create_dataset("auto", data=predictions)

    print(f"\nauto field written to {contractions_h5}")
    if manual_shape is not None:
        print("manual field left untouched.")
    else:
        print("no manual field present, this dataset has not been hand-annotated.")
        print("auto is the only annotation source for this dataset right now.")


if __name__ == "__main__":
    main()
