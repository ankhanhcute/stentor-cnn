from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FAILURES_NPZ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, "outputs", "failures.npz")
OUT_PNG = FAILURES_NPZ.replace(".npz", ".png")

SOURCE_LABEL = {"fp": "UNC, guessed FP", "fn": "UNC, guessed FN", "nan": "UNC, NaN Label"}
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
    unc_tiles, unc_probs, unc_cells, unc_stims, unc_source = (
        data["unc_tiles"], data["unc_probs"], data["unc_cells"], data["unc_stims"], data["unc_source"])

    print(f"FP: {len(fp_probs)} | FN: {len(fn_probs)}")

    n_fp, n_fn, n_unc = len(fp_probs), len(fn_probs), len(unc_probs)
    n_unc_fp = sum(1 for s in unc_source if s == "fp")
    n_unc_fn = sum(1 for s in unc_source if s== "fn")
    n_unc_nan = sum(1 for s in unc_source if s == "nan")
    print(f"confident wrong: FP={n_fp:4d}  FN={n_fn:4d}")
    print(f"uncertain total: {n_unc:4d}  (guessed-FP={n_unc_fp}  guessed-FN={n_unc_fn}  NaN={n_unc_nan})")
    
    false_positives = list(zip(fp_tiles, fp_probs, fp_cells, fp_stims))
    false_negatives = list(zip(fn_tiles, fn_probs, fn_cells, fn_stims))
    uncertain = list(zip(unc_tiles, unc_probs, unc_cells, unc_stims, unc_source))
    plot_failures(false_positives, false_negatives, uncertain, OUT_PNG)
def plot_failures(false_positives, false_negatives, uncertain, out_path, n=10):
    fps = sorted(false_positives, key=lambda x: x[1], reverse=True)[:n]
    fns = sorted(false_negatives,    key=lambda x: x[1])[:n]
    unc = sorted(uncertain, key=lambda x: x[1], reverse=True)[:n]
    actual_n = max(len(fps), len(fns), len(unc), 1)
    fig, axes = plt.subplots(3, actual_n * 2, figsize=(actual_n * 4, 7.5))
    fig.suptitle("Confident Failures (top) + Uncertain Cases (bottom)", fontsize=14)
    

    for i in range(actual_n):
        ax_pre, ax_post = axes[0, i * 2], axes[0, i * 2 + 1]
        if i < len(fps):
            tile, prob, c, k = fps[i]
            ax_pre.imshow(tile[0], cmap='gray'); ax_pre.set_title(f"FP pre\np={prob:.3f} c{c} s{k}", fontsize=7)
            ax_post.imshow(tile[1], cmap='gray'); ax_post.set_title(f"FP post\np={prob:.3f} c{c} s{k}", fontsize=7)
        ax_pre.axis('off'); ax_post.axis('off')
        ax_pre, ax_post = axes[1, i * 2], axes[1, i * 2 + 1]
        if i < len(fns):
            tile, prob, c, k = fns[i]
            ax_pre.imshow(tile[0], cmap='gray'); ax_pre.set_title(f"FN pre\np={prob:.3f} c{c} s{k}", fontsize=7)
            ax_post.imshow(tile[1], cmap='gray'); ax_post.set_title(f"FN post\np={prob:.3f} c{c} s{k}", fontsize=7)
        ax_pre.axis('off'); ax_post.axis('off')

        ax_pre, ax_post = axes[2, i * 2], axes[2, i * 2 + 1]
        if i < len(unc):
            tile, prob, c, k, src = unc[i]
            label = SOURCE_LABEL.get(str(src), "UNC")
            ax_pre.imshow(tile[0], cmap='gray'); ax_pre.set_title(f"{label}\npre p={prob:.3f} c{c} s{k}", fontsize=7)
            ax_post.imshow(tile[1], cmap='gray'); ax_post.set_title(f"{label}\npost p={prob:.3f} c{c} s{k}", fontsize=7)
        ax_pre.axis('off'); ax_post.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
if __name__ == "__main__":
    main()