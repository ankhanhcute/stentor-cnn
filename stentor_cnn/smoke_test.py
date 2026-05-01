"""
smoke_test.py 
-------------
Visualize pre,post, label samples from the loader before feed it to the model, so we 
can eyeball it:
     (a) the reshape produced sensible cell tiles (not rotated/scrambled)]
     (b) channel 0 == pre frame and channel 1 == post frame, 
     (c) label=1 samples actually look like contracted stentor (small dark polka dot
     on the right) but label=0 is unchanged, extended one 

Run from the project root: 
     python stentor_cnn/smoke_test.py 

Output going to stentor_cnn/outputs/smoke_test.png
"""
from __future__ import annotations 

import copy 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import loader

#-----pahts (edit if data folders are named differently)
OUT_DIR = os.path.join(THIS_DIR, "output")
OUT_PNG = os.path.join(THIS_DIR, "smoke_test_v2.png")
if len(sys.argv) !=4:
    print("Usage: python smoke_test.py <tiled.h5> <meta.h5> <contractions.h5>")
    sys.exit(1)


TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
GT_H5 = sys.argv[3]

N_PER_CLASS = 6 #rows of contracted + rows of extended

def main() -> int:
    print("Loading files...")
    #NOTE: respect loader.py signature (meta_h5_path, tiled_h5_path) so keep it (meta, tiled)
    tiles, meta = loader.load_tiles(TILED_H5, META_H5)
    print(f" tiles {tiles.shape} dtype={tiles.dtype} "
          f" min={tiles.min():.3f} max={tiles.max():.3f}")
    print(f"meta {meta}")

    print("Loading manual labels...")
    manual = loader.load_manual_labels(GT_H5)
    print(f"  manual {manual.shape} dtype={manual.dtype} "
          f"pos_frac={np.nanmean(manual):.3f}")

 #all cells for visualization (not training anything)
    all_cells = list(range(meta["num_cells"]))
    ds = loader.StentorPairs(tiles, manual, all_cells) 
    print(f"Dataset size: {len(ds)} samples (pos_frac={ds.positive_fraction():.3f})")

#split it by labels
    pos_indices = [i for i, (_, _, lab) in enumerate(ds.index) if lab == 1.0]
    neg_indices = [i for i, (_, _, lab) in enumerate(ds.index) if lab == 0.0]

    rng = np.random.default_rng(42)
    pos_pick = rng.choice(pos_indices, size=min(N_PER_CLASS, len(pos_indices)),
                replace=False)
    neg_pick = rng.choice(neg_indices, size=min(N_PER_CLASS, len(neg_indices)),
               replace=False)
    
    os.makedirs(OUT_DIR, exist_ok=True)

    n_rows = len(pos_pick) + len(neg_pick)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    def _plot(row:  int, sample_idx: int, tag: str) -> None:
        x, y = ds[sample_idx]
        c, k, _ = ds.index[sample_idx]
        pre = x[0].numpy()
        post = x[1].numpy()
        ax_pre, ax_post = axes[row]
        ax_pre.imshow(pre, cmap="gray", vmin=0, vmax=1)
        ax_post.imshow(post, cmap="gray", vmin=0, vmax=1)
        ax_pre.set_title(f"{tag} cell={c} stim={k} PRE", fontsize=10)
        ax_post.set_title(f"{tag} cell={c} stim={k} POST", fontsize=10)
        for a in (ax_pre, ax_post):
            a.set_xticks([]); a.set_yticks([])

    row = 0 
    for s in pos_pick:
        _plot(row, int(s), tag="LABEL=1 (CONTRACTED)")
        row += 1
    for s in neg_pick:
        _plot(row, int(s), tag="LABEL=0 (NOT CONTRACTED)")
        row += 1

    fig.suptitle(
         f"Smoke test — {len(pos_pick)} positives + {len(neg_pick)} negatives",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\n[OK] saved {OUT_PNG}")
    return 0

if __name__ == "__main__":
    sys.exit(main())