"""
train_fold.py - one fold of k-fold cross-validation
Usage: python cross_val/train_fold.py --fold 0 --k 5
"""
from __future__ import annotations
import sys, os, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
CNN_DIR   = os.path.join(PROJ_ROOT, "stentor_cnn")
PRETRAIN_CKPT = os.path.join(PROJ_ROOT, "stentor_cnn", "checkpoints", "best_model.pt")
sys.path.insert(0, CNN_DIR)

import loader
from model import StentorSequenceModel, count_params

DATA = os.path.join(PROJ_ROOT, "data")

HOLDOUT = {
    '2025_10_20_03_02_39', '2025_11_02_23_30_22', '2025_11_03_20_45_41',
    '2025_10_30_20_47_23', '2025_11_01_17_58_50', '2025_09_22_02_43_12',
    '2025_04_26_23_21_10', '2025_06_12_22_58_40', '2025_04_14_20_26_34',
    '2025_05_25_00_54_10', '2025_10_29_23_54_44', '2025_10_30_20_46_05',
    '2025_11_02_23_29_03', '2025_11_02_23_31_42', '2025_09_22_02_41_52',
    '2025_09_23_19_20_47', '2025_06_13_23_05_12', '2025_05_30_20_11_44',
    '2025_04_26_23_24_08', '2024_12_29_03_27_19', '2024_11_10_11_53_54',
    '2026_05_05_17_39_45',
}

def get_all_recordings():
    names = []
    for f in sorted(os.listdir(DATA)):
        if f.endswith('_tiled.h5'):
            name = f.replace('_tiled.h5', '')
            if name in HOLDOUT:
                continue
            t = os.path.join(DATA, f"{name}_tiled.h5")
            m = os.path.join(DATA, f"{name}_tiled_data.h5")
            c = os.path.join(DATA, f"{name}_contractions.h5")
            if os.path.exists(t) and os.path.exists(m) and os.path.exists(c):
                names.append(name)
    return names

def recording_paths(name):
    return (
        os.path.join(DATA, f"{name}_tiled.h5"),
        os.path.join(DATA, f"{name}_tiled_data.h5"),
        os.path.join(DATA, f"{name}_contractions.h5"),
    )

BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3

def compute_metrics(all_logits, all_labels, threshold=0.5):
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    probs  = torch.sigmoid(logits)
    pred   = (probs > threshold).float()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    acc       = (tp + tn) / max(tp + fp + fn + tn, 1)
    return dict(acc=acc, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)

@torch.no_grad()
def flag_uncertain(model, datasets, device, threshold=0.5, margin=0.2):
    model.eval()
    uncertain = []
    lo, hi = 0.5 - margin, 0.5 + margin
    for ds in datasets:
        for i, cell_idx in enumerate(ds.index):
            seq, labels = ds[i]
            probs = torch.sigmoid(model(seq.unsqueeze(0).to(device))).squeeze().cpu()
            for k in range(len(labels)):
                lab = labels[k].item()
                if lab == -1:
                    continue
                prob = probs[k].item()
                if lo < prob < hi:
                    uncertain.append({
                        "cell": int(cell_idx), "stimulus": int(k),
                        "prob": round(prob, 4), "true_label": int(lab),
                        "pred": int(prob >= threshold),
                        "correct": int((prob >= threshold) == bool(lab)),
                    })
    return uncertain

def train_one_epoch(model, dl, loss_fn, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_logits, all_labels = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        mask   = (y != -1)
        loss   = loss_fn(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
        all_logits.append(logits[mask].detach())
        all_labels.append(y[mask].detach())
    return total_loss / max(n_batches, 1), compute_metrics(all_logits, all_labels)

@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_logits, all_labels = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device).float()
        logits = model(x).squeeze(-1)
        mask   = (y != -1)
        loss   = loss_fn(logits[mask], y[mask])
        total_loss += loss.item(); n_batches += 1
        all_logits.append(logits[mask])
        all_labels.append(y[mask])
    return total_loss / max(n_batches, 1), compute_metrics(all_logits, all_labels), all_logits, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",   type=int, required=True)
    parser.add_argument("--k",      type=int, default=5)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ALL_RECORDINGS = get_all_recordings()
    print(f"[fold {args.fold}/{args.k}] device: {device}  total datasets: {len(ALL_RECORDINGS)}")

    indices = list(range(len(ALL_RECORDINGS)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    fold_size = len(ALL_RECORDINGS) // args.k
    val_idx   = indices[args.fold * fold_size : (args.fold + 1) * fold_size]
    train_idx = [i for i in indices if i not in val_idx]

    train_names = [ALL_RECORDINGS[i] for i in train_idx]
    val_names   = [ALL_RECORDINGS[i] for i in val_idx]
    print(f"  train ({len(train_names)})  val ({len(val_names)}): {val_names}")

    train_datasets, val_datasets = [], []
    for name in train_names:
        t, m, g = recording_paths(name)
        tiles, meta = loader.load_tiles(t, m)
        manual = loader.load_manual_labels(g)
        tr, va, _ = loader.make_cell_disjoint_split(meta["num_cells"])
        train_datasets.append(loader.StentorPairs(tiles, manual, tr, tiled_h5_path=t))
    for name in val_names:
        t, m, g = recording_paths(name)
        tiles, meta = loader.load_tiles(t, m)
        manual = loader.load_manual_labels(g)
        all_cells = list(range(meta["num_cells"]))
        val_datasets.append(loader.StentorPairs(tiles, manual, all_cells, tiled_h5_path=t))

    ds_train = ConcatDataset(train_datasets)
    ds_val   = ConcatDataset(val_datasets)
    pos_frac  = sum(ds.positive_fraction() * len(ds) for ds in train_datasets) / max(len(ds_train), 1)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = StentorSequenceModel(dropout=DROPOUT).to(device)
    if os.path.exists(PRETRAIN_CKPT):
        model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=device, weights_only=True))
        print(f" params: {count_params(model):,} (fine_tuning from {PRETRAIN_CKPT})")
    else:
        print(f" params: {count_params(model):,} (training from scratch)")
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    print(f"  pos_weight: {pos_weight.item():.2f}")
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.7, patience=15)

    ckpt_dir    = os.path.join(THIS_DIR, "checkpoints", f"fold_{args.fold}")
    results_dir = os.path.join(PROJ_ROOT, "results", "fold_results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    best_val_f1 = -1.0
    t_start     = time.time()
    print(f"\n{'epoch':>5}  {'tr_loss':>8}  {'tr_f1':>6}  {'v_loss':>8}  {'v_f1':>6}  {'best':>4}  {'lr':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m        = train_one_epoch(model, dl_train, loss_fn, optimizer, device)
        vl_loss, vl_m, _, _  = evaluate(model, dl_val, loss_fn, device)
        improved = ""
        if vl_m["f1"] > best_val_f1:
            best_val_f1 = vl_m["f1"]
            torch.save(model.state_dict(), best_ckpt)
            improved = " *"
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_m['f1']:6.3f}  "
              f"{vl_loss:8.4f}  {vl_m['f1']:6.3f}  {improved:>4}  {lr_now:.1e}")
        scheduler.step(vl_m["f1"])

    elapsed = time.time() - t_start
    print(f"\nFold {args.fold} done in {elapsed:.1f}s  best_val_f1={best_val_f1:.4f}")

    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    _, _, val_logits, val_labels = evaluate(model, dl_val, loss_fn, device)
    best_thresh = max([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
                      key=lambda t: compute_metrics(val_logits, val_labels, t)["f1"])
    best_m = compute_metrics(val_logits, val_labels, best_thresh)

    print(f"\n--- Fold {args.fold} val @ thresh={best_thresh} ---")
    print(f"  acc={best_m['acc']:.3f}  prec={best_m['precision']:.3f}  "
          f"rec={best_m['recall']:.3f}  f1={best_m['f1']:.3f}")
    print(f"  TP={best_m['tp']}  FP={best_m['fp']}  FN={best_m['fn']}  TN={best_m['tn']}")

    uncertain = flag_uncertain(model, val_datasets, device, threshold=best_thresh, margin=args.margin)
    print(f"\n  {len(uncertain)} uncertain stimuli flagged (margin={args.margin})")

    results = {
        "fold": args.fold, "k": args.k,
        "val_datasets": val_names, "train_datasets": train_names,
        "best_threshold": best_thresh, "metrics": best_m,
        "uncertain_count": len(uncertain), "uncertain_stimuli": uncertain,
        "elapsed_s": round(elapsed, 1),
    }
    out_path = os.path.join(results_dir, f"fold_{args.fold}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results saved → {out_path}")

if __name__ == "__main__":
    main()
