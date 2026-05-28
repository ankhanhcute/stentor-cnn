"""
evaluate.py
-----------
Run a pretrained model on a completely unseen dataset.

Usage:
    python evaluate.py <tiled.h5> <meta.h5> <contractions.h5> [checkpoint.pt]

Example:
    python evaluate.py \
        ../tiles/2024_12_06_05_07_49_tiled.h5 \
        ../meta/2024_12_06_05_07_49_tiled_data.h5 \
        ../contraction/2024_12_06_05_07_49_contractions.h5
"""
from __future__ import annotations
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import loader
from model import StentorCNN, count_params

# ---- Config ----
BATCH_SIZE = 32
DROPOUT = 0.3
CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best_model.pt")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

if len(sys.argv) < 4:
    print("Usage: python evaluate.py <tiled.h5> <meta.h5> <contractions.h5> [checkpoint.pt]")
    sys.exit(1)

TILED_H5 = sys.argv[1]
META_H5  = sys.argv[2]
GT_H5    = sys.argv[3]
if len(sys.argv) > 4:
    CHECKPOINT = sys.argv[4]

# ---- Metrics ----
def compute_metrics(all_logits, all_labels, threshold=0.5):
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    probs  = torch.sigmoid(logits)
    pred   = (probs > threshold).float()

    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()

    acc       = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(acc=acc, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)

# ---- Evaluate ----
@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_logits, all_labels = [], []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)
        logits = model(x)
        loss   = loss_fn(logits, y)
        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits)
        all_labels.append(y)
    return total_loss / max(n_batches, 1), all_logits, all_labels

# ---- Save failures ----
@torch.no_grad()
def save_failures(model, ds_list, device, threshold, dataset_name):
    model.eval()
    fp_tiles, fp_probs, fp_cells, fp_stims = [], [], [], []
    fn_tiles, fn_probs, fn_cells, fn_stims = [], [], [], []
    for ds in ds_list:
        for i, (c, k, _) in enumerate(ds.index):
            tile, label = ds[i]
            prob  = torch.sigmoid(model(tile.unsqueeze(0).to(device))).item()
            pred  = 1 if prob >= threshold else 0
            truth = int(label.item())
            if pred == 1 and truth == 0:
                fp_tiles.append(tile.numpy()); fp_probs.append(prob)
                fp_cells.append(c);            fp_stims.append(k)
            elif pred == 0 and truth == 1:
                fn_tiles.append(tile.numpy()); fn_probs.append(prob)
                fn_cells.append(c);            fn_stims.append(k)
    _empty   = np.zeros((0, 2, 1, 1), dtype=np.float32)
    out_path = os.path.join(OUT_DIR, f"failures_{dataset_name}.npz")
    np.savez(out_path,
        fp_tiles=np.array(fp_tiles, dtype=np.float32) if fp_tiles else _empty,
        fp_probs=np.array(fp_probs), fp_cells=np.array(fp_cells), fp_stims=np.array(fp_stims),
        fn_tiles=np.array(fn_tiles, dtype=np.float32) if fn_tiles else _empty,
        fn_probs=np.array(fn_probs), fn_cells=np.array(fn_cells), fn_stims=np.array(fn_stims),
    )
    print(f"  Failures saved → {out_path}  (FP={len(fp_tiles)}  FN={len(fn_tiles)})")

# ---- Main ----
def main():
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else "cpu"
    )
    print(f"device: {device}")
    print(f"checkpoint: {CHECKPOINT}")
    print(f"dataset: {TILED_H5}\n")

    # load ALL cells as test (no train/val split — this is unseen data)
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    manual      = loader.load_manual_labels(GT_H5)
    all_cells   = list(range(meta["num_cells"]))
    ds          = loader.StentorPairs(tiles, manual, all_cells, tiled_h5_path=TILED_H5)
    dl          = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_frac    = ds.positive_fraction()
    pos_weight  = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    loss_fn     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"  total samples : {len(ds)}")
    print(f"  pos_weight    : {pos_weight.item():.2f}")
    print(f"  positive frac : {pos_frac:.3f}\n")

    # load model
    model = StentorCNN(in_channels=2, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    print(f"  model params: {count_params(model):,}")

    avg_loss, all_logits, all_labels = evaluate(model, dl, loss_fn, device)

    # find best threshold
    best_thresh = max([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
                      key=lambda t: compute_metrics(all_logits, all_labels, t)['f1'])

    m      = compute_metrics(all_logits, all_labels, threshold=0.5)
    best_m = compute_metrics(all_logits, all_labels, threshold=best_thresh)

    print(f"--- Results (threshold=0.5) ---")
    print(f"  loss={avg_loss:.4f}  acc={m['acc']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1']:.3f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    print(f"\n--- Results at best threshold ({best_thresh}) ---")
    print(f"  acc={best_m['acc']:.3f}  prec={best_m['precision']:.3f}  rec={best_m['recall']:.3f}  f1={best_m['f1']:.3f}")
    print(f"  TP={best_m['tp']}  FP={best_m['fp']}  FN={best_m['fn']}  TN={best_m['tn']}")

    print(f"\n--- Threshold Sweep ---")
    print(f"{'threshold':>10}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'acc':>6}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        mm = compute_metrics(all_logits, all_labels, threshold=t)
        print(f"{t:>10.2f}  {mm['precision']:>6.3f}  {mm['recall']:>6.3f}  {mm['f1']:>6.3f}  {mm['acc']:>6.3f}")

    # save failures
    dataset_name = os.path.basename(TILED_H5).replace("_tiled.h5", "")
    print(f"\n--- Saving failures ---")
    save_failures(model, [ds], device, best_thresh, dataset_name)

if __name__ == "__main__":
    main()