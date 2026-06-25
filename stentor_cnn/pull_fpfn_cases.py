#!/usr/bin/env python3
"""
pull_fpfn_cases.py
-------------------
Pull and visualize the FP/FN cases for a dataset, split by whether the
uncertainty flag caught them or missed them.

Usage:
    python pull_fpfn_cases.py <dataset_name>
Example:
    python pull_fpfn_cases.py 2024_12_07_01_56_50
"""
import sys, os, json
import numpy as np
import h5py
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

DATA_DIR = os.path.join(THIS_DIR, "..", "data")
OUT_DIR  = os.path.join(THIS_DIR, "outputs")

def main(dataset_name):
    pred_path = os.path.join(OUT_DIR, f"predictions_{dataset_name}.json")
    gt_path   = os.path.join(DATA_DIR, f"{dataset_name}_contractions.h5")
    tiled_path= os.path.join(DATA_DIR, f"{dataset_name}_tiled.h5")

    with open(pred_path) as f:
        preds = json.load(f)

    with h5py.File(gt_path, "r") as f:
        manual = f["manual"][:]   # shape (num_cells, total_stims)

    pred_map = {(p["cell"], p["stimulus"]): p for p in preds}

    caught = []   # FP/FN AND flagged uncertain
    missed = []   # FP/FN, NOT flagged uncertain (confidently wrong)
    false_alarm = []  # flagged uncertain, NOT actually FP/FN
    nan_unclear = []  # no real ground truth to grade against at all

    for (c, k), p in pred_map.items():
        lab = manual[c, k]
        if np.isnan(lab) or lab == -1:
            nan_unclear.append((c, k, p["probability"], lab, p["prediction"]))
            continue
        truth = int(lab)
        pred = p["prediction"]
        is_uncertain = pred is None
        is_wrong = (pred is not None) and (pred != truth)

        if is_uncertain:
            # need to know what the "would-be" hard call was to call it caught vs false-alarm
            # use probability vs 0.5 as the implied hard call
            implied = 1 if p["probability"] >= 0.5 else 0
            if implied != truth:
                caught.append((c, k, p["probability"], truth, implied))
            else:
                false_alarm.append((c, k, p["probability"], truth, implied))
        elif is_wrong:
            missed.append((c, k, p["probability"], truth, pred))

    print(f"caught (uncertain, wrong implied):     {len(caught)}")
    print(f"missed (confident, wrong):              {len(missed)}")
    print(f"false_alarm (uncertain, correct implied): {len(false_alarm)}")
    print(f"nan_unclear (no ground truth, excluded):  {len(nan_unclear)}")
    print()
    for label, group in [("CAUGHT", caught), ("MISSED", missed), ("FALSE_ALARM", false_alarm)]:
        print(f"--- {label} ---")
        for c, k, prob, truth, pred in group:
            print(f"  cell={c:3d} stim={k:3d}  prob={prob:.4f}  truth={truth}  pred={pred}")
        print()
    print(f"--- NAN_UNCLEAR ---")
    for c, k, prob, lab, pred in nan_unclear:
        print(f"  cell={c:3d} stim={k:3d}  prob={prob:.4f}  manual_label={lab}  pred={pred}")
    print()

    # ---- render tile grids ----
    with h5py.File(tiled_path, "r") as f:
        # adjust key name if your tiled.h5 uses something other than "tiles"
        tile_keys = list(f.keys())
        print(f"tiled.h5 keys: {tile_keys}")

    plot_group(tiled_path, caught, f"caught_{dataset_name}.png", "CAUGHT (uncertain, wrong)")
    plot_group(tiled_path, missed, f"missed_{dataset_name}.png", "MISSED (confident, wrong)")
    plot_group(tiled_path, false_alarm, f"false_alarm_{dataset_name}.png", "FALSE ALARM (uncertain, correct)")


def plot_group(tiled_path, group, out_name, title):
    if not group:
        print(f"(skipping {out_name}, empty group)")
        return
    n = len(group)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 4, nrows * 2.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    with h5py.File(tiled_path, "r") as f:
        for idx, (c, k, prob, truth, pred) in enumerate(group):
            row, col = idx // ncols, idx % ncols
            # adjust this indexing to match your actual tiled.h5 structure
            # assumes f["tiles"] shape (num_cells, total_stims, 2, H, W) -> [pre, post]
            try:
                pair = f["tiles"][c, k]   # shape (2, H, W)
                pre, post = pair[0], pair[1]
            except Exception as e:
                axes[row, col*2].text(0.5, 0.5, f"load err\n{e}", ha="center")
                axes[row, col*2+1].axis("off")
                continue
            axes[row, col*2].imshow(pre, cmap="gray")
            axes[row, col*2].set_title(f"c{c}s{k} PRE\np={prob:.2f} t={truth} pr={pred}", fontsize=7)
            axes[row, col*2].axis("off")
            axes[row, col*2+1].imshow(post, cmap="gray")
            axes[row, col*2+1].set_title("POST", fontsize=7)
            axes[row, col*2+1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pull_fpfn_cases.py <dataset_name>")
        sys.exit(1)
    main(sys.argv[1])