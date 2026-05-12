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
    (3, 10, "cell3_pre_stim5"),
    (3, 11, "cell3_post_stim5"),
    (5, 0, "cell5_pre_stim0"),
    (5, 1, "cell5_post_stim0"),
    (8, 40, "cell8_pre_stim20"),
    (8, 41, "cell8_post_stim20"),
    (15, 60, "cell15_pre_stim30"),
    (15, 61, "cell15_post_stim30"),
]

fig, axes = plt.subplots(len(test_cases), 3, figsize=(10, 3 * len(test_cases)))
for 

fig.tight_layout()
os.makedirs("stentor_cnn/outputs", exist_ok=True)
fig.savefig("stentor_cnn/outputs/_v6.png", dpi=120, bbox_inches="tight")
print(f"Saved cellpose_test.png — open it and check if the masks look right")
