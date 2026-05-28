from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FAILURES_NPZ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, "outputs", "failures.npz")
OUT_PNG = FAILURES_NPZ.replace(".npz", ".png")

def main():
    # load what train.py saved
    if not os.path.exists(FAILURES_NPZ):
        print(f"ERROR: {FAILURES_NPZ} not found. Run train.py first!")
        sys.exit(1)

    data = np.load(FAILURES_NPZ)
    fp_tiles = data["fp_tiles"]
    fp_probs = data["fp_probs"]
    fp_cells = data["fp_cells"]
    fp_stims = data["fp_stims"]
    fn_tiles = data["fn_tiles"]
    fn_probs = data["fn_probs"]
    fn_cells = data["fn_cells"]
    fn_stims = data["fn_stims"]

    print(f"FP: {len(fp_probs)} | FN: {len(fn_probs)}")

    # build into list format for plot_failures
    false_positives = list(zip(fp_tiles, fp_probs, fp_cells, fp_stims))
    false_negatives = list(zip(fn_tiles, fn_probs, fn_cells, fn_stims))

    plot_failures(false_positives, false_negatives, OUT_PNG)

def plot_failures(false_positives, false_negatives, out_path, n=10):
    fps = sorted(false_positives, key=lambda x: x[1], reverse=True)[:n]
    fns = sorted(false_negatives,    key=lambda x: x[1])[:n]

    actual_n = max(len(fps), len(fns), 1)
    fig, axes = plt.subplots(2, actual_n * 2, figsize=(actual_n * 4, 5))
    fig.suptitle("Failure Cases", fontsize=14)

    for i in range(actual_n):
        for row, cases, label in [(0, fps, "FP"), (1, fns, "FN")]:
            ax_pre  = axes[row, i * 2]
            ax_post = axes[row, i * 2 + 1]
            if i < len(cases):
                tile, prob, c, k = cases[i]
                ax_pre.imshow(tile[0], cmap='gray')
                ax_pre.set_title(f"{label} pre\np={prob:.2f} c{c} s{k}", fontsize=7)
                ax_post.imshow(tile[1], cmap='gray')
                ax_post.set_title(f"{label} post\np={prob:.2f} c{c} s{k}", fontsize=7)
            ax_pre.axis('off')
            ax_post.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()