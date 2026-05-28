"""
check_masks.py
--------------
python check_masks.py <tiled.h5> <meta.h5> <contractions.h5>
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import loader

TILED_H5 = sys.argv[1]
META_H5  = sys.argv[2]
GT_H5    = sys.argv[3]
OUT_PNG  = os.path.join(THIS_DIR, "outputs", "check_masks2.png")

def main():
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    manual = loader.load_manual_labels(GT_H5)
    num_cells = meta["num_cells"]

    # use StentorPairs so holdfasts load exactly like training
    ds = loader.StentorPairs(tiles, manual, list(range(num_cells)), tiled_h5_path=TILED_H5)

    cols = 4
    rows = (num_cells + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for c in range(num_cells):
        ax = axes[c]
        frame = tiles[c, :, :, 0]
        ax.imshow(frame, cmap='gray')
    
    # show segmentation mask as overlay
        mask = ds.cell_masks[c]
        overlay = np.zeros((*frame.shape, 4))
        overlay[mask] = [1, 0, 0, 0.3]  # red transparent overlay
        ax.imshow(overlay)
    
    # still show holdfast point
        cy, cx = ds.holdfasts[c]
        ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        ax.set_title(f"c{c}", fontsize=7)
        ax.axis('off')

    for i in range(num_cells, len(axes)):
        axes[i].axis('off')

    plt.suptitle("red + = holdfast, circle = mask boundary", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved → {OUT_PNG}")

if __name__ == "__main__":
    main()