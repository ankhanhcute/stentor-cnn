"""
train.py
--------
Train StentorCNN on raw pre/post frame pairs.
Usage (from project root):
    python stentor_cnn/train.py
All paths and hyperparameters are set as constants at the top of the file.
Edit them directly — no argparse complexity for a single-dataset project.
"""
from __future__ import annotations 
import sys
import os

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import loader
from model import StentorCNN, count_params

#------Configuration-------
THIS_DIR = (os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

sys.path.insert(0, THIS_DIR)

if len(sys.argv) < 4 or (len(sys.argv) - 1) % 3 !=0 :
    print(f"Usage: python train.py <tiled.h5> <meta.h5> <contractions.h5>")
    sys.exit(1)

recordings = []
for i in range(1, len(sys.argv), 3):
    recordings.append((sys.argv[i], sys.argv[i+1], sys.argv[i+2]))

TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
GT_H5 = sys.argv[3]


#-------Hyperparameters------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
DROPOUT = 0.3
SEED = 42

#--------Cells Split---------
VAL_CELLS = None # e.g. [16, 17, 18] to fix specific val cells
TEST_CELLS = None # e.g. [19, 20, 21] to fix specific test cells

# Output 
CHECKPOINT_DIR = os.path.join(THIS_DIR, "checkpoints")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

#---------------Metrics----------------------

def compute_metrics(
    all_logits: list[torch.Tensor], 
    all_labels: list[torch.Tensor], 
    threshold: float = 0.5, 
) -> dict:
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()

    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall /  max(precision + recall, 1e-9)

    return dict(acc=acc, precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, 
    fn=fn, tn=tn)
#--------Train one epoch--------

def train_one_epoch(
    model: nn.Module, 
    dl: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad() #clear gradient from the prev batch
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach())
        all_labels.append(y.detach())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_logits, all_labels)
    return avg_loss, metrics

#-----------Evaluate--------
@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dl: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches = 0 
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits)
        all_labels.append(y)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_logits, all_labels)
    return avg_loss, metrics, all_logits, all_labels

#---------Plot for the training curves?_---------

def plot_curves(history: dict, path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    #Loss
    axes[0].plot(epochs, history["train_loss"], label='train')
    axes[0].plot(epochs, history["val_loss"], label='val')
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    #Accuracy 
    axes[1].plot(epochs, history["train_acc"], label='train')
    axes[1].plot(epochs, history["val_acc"], label='val')
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0.5, 1.0)

    #F1
    axes[2].plot(epochs, history["train_f1"], label='train')
    axes[2].plot(epochs, history["val_f1"], label='val')
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("f1")
    axes[2].set_title("F1 Score")
    axes[2].legend()
    axes[2].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  curves saved to {path}")

#-------Main-------
def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = ( 
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device: {device}")

    print("Loading tiles...")
    train_datasets  = []
    val_datasets = []
    test_datasets = []

    #-----Split------
    for tiled_path, meta_path, gt_path in recordings:
        tiles, meta = loader.load_tiles(tiled_path, meta_path)
        manual = loader.load_manual_labels(gt_path)
        tr, va, te = loader.make_cell_disjoint_split(meta["num_cells"], val_cells=VAL_CELLS, test_cells = TEST_CELLS)
        train_datasets.append(loader.StentorPairs(tiles, manual, tr, tiled_h5_path=tiled_path))
        val_datasets.append(loader.StentorPairs(tiles, manual, va, tiled_h5_path=tiled_path))
        test_datasets.append(loader.StentorPairs(tiles, manual, te, tiled_h5_path=tiled_path))
    
    ds_train = ConcatDataset(train_datasets)
    ds_val = ConcatDataset(val_datasets)
    ds_test = ConcatDataset(test_datasets)

    total_pos  = sum(ds.positive_fraction() * len(ds) for ds in train_datasets)
    total_samp = sum(len(ds) for ds in train_datasets)
    pos_frac   = total_pos / max(total_samp, 1)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)
    dl_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)
    # --- Model ---
    model = StentorCNN(in_channels=2, dropout=DROPOUT).to(device)
    print(f"  model params: {count_params(model):,}")
    # --- Loss with pos_weight to handle class imbalance ---
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)],
                              device=device)
    print(f"  pos_weight: {pos_weight.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # --- Optimizer + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7)

    # --- Training loop ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
    }
    best_val_f1 = -1.0
    print(f"\n{'epoch':>5}  {'tr_loss':>8}  {'tr_acc':>6}  {'tr_f1':>6}  "
          f"{'v_loss':>8}  {'v_acc':>6}  {'v_f1':>6}  {'best':>4}  {'lr':>8}")
    print("-" * 72)
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_m = train_one_epoch(model, dl_train, loss_fn, optimizer, device)
        vl_loss, vl_m, _, _= evaluate(model, dl_val, loss_fn, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_m["acc"])
        history["val_acc"].append(vl_m["acc"])
        history["train_f1"].append(tr_m["f1"])
        history["val_f1"].append(vl_m["f1"])
        improved = ""
        if vl_m["f1"] > best_val_f1:
            best_val_f1 = vl_m["f1"]
            torch.save(model.state_dict(), best_ckpt)
            improved = " *"
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_m['acc']:6.3f}  {tr_m['f1']:6.3f}  "
              f"{vl_loss:8.4f}  {vl_m['acc']:6.3f}  {vl_m['f1']:6.3f}  "
              f"{improved:>4}  {lr_now:.1e}")
        scheduler.step(vl_m["f1"])
    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed:.1f}s  ({elapsed/EPOCHS:.1f}s/epoch)")
    print(f"Best val F1: {best_val_f1:.4f}")
    # --- Final test evaluation ---
    print("\n--- Test set (using best checkpoint) ---")
    model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                     weights_only=True))
    te_loss, te_m, te_logits, te_labels = evaluate(model, dl_test, loss_fn, device)
    print(f"  loss={te_loss:.4f}  acc={te_m['acc']:.3f}  "
          f"prec={te_m['precision']:.3f}  rec={te_m['recall']:.3f}  "
          f"f1={te_m['f1']:.3f}")
    print(f"  TP={te_m['tp']}  FP={te_m['fp']}  FN={te_m['fn']}  TN={te_m['tn']}")
    print("\n--- Threshold Sweep ---")
    print(f"{'threshold':>10}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'acc':>6}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        m = compute_metrics(te_logits, te_labels, threshold=thresh)
        print(f"{thresh:>10.2f}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['acc']:>6.3f}")
    best_thresh = max([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8], 
                  key=lambda t: compute_metrics(te_logits, te_labels, threshold=t)['precision']
                  if compute_metrics(te_logits, te_labels, threshold=t)['recall'] > 0.85 
                  else 0)
    print(f"\nBest threshold for precision (recall > 0.85): {best_thresh}")
    # --- Save curves ---
    plot_curves(history, os.path.join(OUT_DIR, "training_curves.png"))
    return 0
if __name__ == "__main__":
    sys.exit(main())
