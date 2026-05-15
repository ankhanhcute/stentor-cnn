from __future__ import annotations 
import torch 
import torch.nn as nn
from model import StentorCNN
import sys 
import os 
import numpy as np
import matplotlib.pyplot as plt
from find_holdfast import find_holdfast

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import loader

OUT_DIR = os.path.join(THIS_DIR, 'outputs')
OUT_PNG = os.path.join(THIS_DIR, 'visualize_failure5.png')

if len(sys.argv) != 5:
    print("Usage: python visualize_failures.py <tile.h5> <meta.h5> <contractions.h5> <checkpoint.pt>")
    sys.exit(1)

TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
GT_H5 = sys.argv[3]
CKPT = sys.argv[4]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # load data
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    print("tiles shape:", tiles.shape)
    manual = loader.load_manual_labels(GT_H5)
    train_cells, val_cells, test_cells = loader.make_cell_disjoint_split(
    meta["num_cells"], seed=42
)
    print(f"Test cells: {test_cells}")

    # compute holdfasts (same as training)
    print("Computing holdfasts...")
    holdfasts = []
    cache_path = TILED_H5.replace(".h5", "_holdfasts.npy")
    if os.path.exists(cache_path):
        holdfasts = list(np.load(cache_path))
        print("Loaded holdfasts from cache")
    else:
        for c in range(tiles.shape[0]):
            result = find_holdfast(tiles[c], tiles.shape[1])
            holdfasts.append(result['holdfast'])
        np.save(cache_path, np.array(holdfasts))
        print("Computed and cached holdfasts")

    # load model
    model = StentorCNN(in_channels=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    # run inference on all cells
    num_cells = tiles.shape[0]
    total_stims = manual.shape[1]
    false_positives = []
    false_negatives = []

    with torch.no_grad():
        for c in test_cells:
            cy, cx = holdfasts[c]
            h, w = tiles.shape[1], tiles.shape[2]
            mask = loader.make_circular_mask(h, w, cy, cx, r=40)

            for k in range(total_stims):
                if k + 1 >= tiles.shape[-1] // 2:
                    continue

                lab = manual[c, k]
                if np.isnan(lab):
                    continue

                # match training exactly
                pre  = tiles[c, :, :, 2 * k]     * mask
                post = tiles[c, :, :, 2 * k + 1] * mask
                tile = np.stack([pre, post], axis=0)  # (2, H, W)

                tile_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(tile_tensor)).item()
                pred = 1 if prob >= 0.5 else 0
                truth = int(lab)

                if pred == 1 and truth == 0:
                    false_positives.append((tile, prob, c, k))
                elif pred == 0 and truth == 1:
                    false_negatives.append((tile, prob, c, k))

    print(f"FP: {len(false_positives)} | FN: {len(false_negatives)}")
    plot_failures(false_positives, false_negatives, OUT_PNG)

def plot_failures(false_positives, false_negatives, out_path, n=10):
    fps = sorted(false_positives, key=lambda x: x[1], reverse=True)[:n]
    fns = sorted(false_negatives, key=lambda x: x[1])[:n]

    actual_n = max(len(fps), len(fns), 1)
    fig, axes = plt.subplots(2, actual_n, figsize=(actual_n * 2, 5))
    fig.suptitle("Failure Cases", fontsize=14)

    for i in range(actual_n):
        for row, cases, label in [(0, fps, "FP"), (1, fns, "FN")]:
            ax = axes[row, i]
            if i < len(cases):
                tile, prob, c, k = cases[i]
                ax.imshow(tile[0], cmap='gray')
                ax.set_title(f"{label} p={prob:.2f}\nc{c} s{k}", fontsize=7)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()