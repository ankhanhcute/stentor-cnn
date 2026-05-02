import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stentor_cnn"))
import numpy as np 
import matplotlib.pyplot as plt
import loader 
from cellpose import models
from scipy import ndimage

if len(sys.argv) != 3:
    print("Usage: python smoke_test.py <tiled.h5> <meta.h5>")
    sys.exit(1)

TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]

tiles, meta = loader.load_tiles(TILED_H5, META_H5)

# 1. CELLPOSE 4 API: Only models.CellposeModel is valid.
# Use "cyto3" or "cpsam" (SAM model) for best results.
model = models.CellposeModel(model_type="cyto3", gpu=False)

test_cases = [
    (0, 0, "cell0_pre_stim0"),
    (0, 1, "cell0_post_stim0"),
    (5, 0, "cell5_pre_stim0"),
    (5, 1, "cell5_post_stim0"),
    (10, 20, "cell10_pre_stim10"),
    (10, 21, "cell10_post_stim10"),
]

fig, axes = plt.subplots(len(test_cases), 3, figsize=(10, 3 * len(test_cases)))

for row, (cell, frame, name) in enumerate(test_cases):
    img = tiles[cell, :, :, frame]
    img_uint8 = (img * 255).astype(np.uint8)

    # 2. EVAL: Evaluate directly on the image with diameter=None. 
    # Use flow_threshold=0.8 to prevent Stentor over-splitting.
    masks, flows, styles = model.eval(
        img_uint8, 
        diameter=None, 
        channels=[0, 0],
        flow_threshold=0.8,
        cellprob_threshold=-1.0
    )

    center = (75, 75)
    best_mask_id = None
    best_dist = float('inf')

    # 3. Filter by size and centroid distance
    for i in range(1, masks.max() + 1):
        ys, xs = np.where(masks == i)
        area_i = len(ys)
        print(f"  [{name}] mask {i}: area={area_i}px")

        # Stentors are large, so skip tiny debris and background noise
        if area_i < 80 or area_i > 500:  
            continue

        cy, cx = ys.mean(), xs.mean()
        dist = (cy - center[0])**2 + (cx - center[1])**2

        if dist < best_dist:
            best_dist = dist
            best_mask_id = i

    # 4. Isolate the Stentor mask
    clean_mask = (masks == best_mask_id).astype(np.uint8) if best_mask_id else np.zeros_like(masks)
    area = int(np.sum(clean_mask)) if best_mask_id else 0

    # Plot results
    axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{name}", fontsize=9)
    
    # Plot only the single selected Stentor mask
    axes[row, 1].imshow(clean_mask, cmap="gray")
    axes[row, 1].set_title(f"selected cell area={area}px", fontsize=9)
    
    # Plot original with contours
    axes[row, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
    if best_mask_id:
        boundary = ndimage.binary_dilation(clean_mask) & ~clean_mask
        axes[row, 2].contour(boundary, colors=['#FF0000'], linewidths=1.5)
    
    axes[row, 2].set_title(f"contour area={area}px", fontsize=9)
    
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
os.makedirs("stentor_cnn/outputs", exist_ok=True)
fig.savefig("stentor_cnn/outputs/_v5.png", dpi=120, bbox_inches="tight")
print(f"Saved cellpose_test.png — open it and check if the masks look right")
