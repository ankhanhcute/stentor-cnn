# Stentor CNN

A convolutional neural network pipeline for detecting contraction events in *Stentor coeruleus* (a single-celled ciliate) from paired pre- and post-stimulus microscopy frames.

---

## Overview

*Stentor coeruleus* contracts in response to mechanical or chemical stimuli. Over repeated stimulation, cells habituate and gradually stop responding. This project automates contraction detection by training a binary classifier on paired image tiles — one frame captured just before a stimulus and one immediately after. The model predicts whether the cell contracted.


---

## Repository Structure

```
.
├── stentor_cnn/
│   ├── model.py                # StentorCNN architecture
│   ├── loader.py               # HDF5 loading, Dataset class, holdfast masking, median filter
│   ├── train.py                # Training loop, cell-disjoint splits, per-dataset failure saving
│   ├── evaluate.py             # Cross-dataset evaluation on unseen recordings (no training)
│   ├── find_holdfast.py        # Holdfast detection (segmentation + skeleton fallback)
│   ├── visualize_failures.py   # Plot false positives and false negatives per dataset
│   ├── export_onnx.py          # ONNX model export
│   ├── smoke_test.py           # Visual sanity check of data loading
│   ├── inspect_data.py         # Print shapes, label distribution, NaN counts
│   ├── data_integrity.py       # Automated data validation checks
│   └── requirements.txt        # Python dependencies
│
├── tiles/                      # Pre-processed cell tile HDF5 files (*_tiled.h5)
├── meta/                       # Metadata HDF5 files (*_tiled_data.h5)
├── contraction/                # Ground-truth label HDF5 files (*_contractions.h5)
└── auto_annotate_v3.jl         # Julia preprocessing pipeline (raw video → HDF5 tiles)
```

---

## Data Format

Each recording produces three HDF5 files:

| File | Contents |
|---|---|
| `*_tiled.h5` | Cell tile images, one dataset per trial (`tiled_frames_trial_0`, `tiled_frames_trial_1`, …) |
| `*_tiled_data.h5` | Metadata: `crop_size`, `num_cells`, `num_trials`, `row_num`, `col_num`, `cell_locs` |
| `*_contractions.h5` | Manual labels in dataset `manual`, shape `(num_cells, num_stims)`, values `0 / 1 / NaN` |

**Frame convention:** for stimulus `k`, the pre-stimulus frame is at index `2k` and the post-stimulus frame is at `2k+1`. The loader stacks these into a 2-channel input of shape `(2, 150, 150)`.

**NaN labels** are excluded from training and evaluation. but still keep in the data just stay away from training

---

## Pipeline Overview

The full workflow from raw video to a trained classifier:

```
Raw video (.mp4)
      │
      ▼
1. Preprocessing  (auto_annotate_v3.jl)
   Detects individual Stentor cells, crops each one into a 150×150 tile,
   and saves paired frames (pre + post per stimulus) into HDF5.
   Produces: *_tiled.h5  *_tiled_data.h5  *_contractions.h5
      │
      ▼
2. Inspect data  (inspect_data.py)
   Prints shapes, dtypes, label distribution, and NaN counts.
   Run this first on any new recording.
      │
      ▼
3. Validate & smoke test  (data_integrity.py, smoke_test.py)
   data_integrity.py runs automated checks on tile normalization,
   frame pairing, and label format.
   smoke_test.py saves a visual grid of pre/post pairs so you can
   manually verify the tiles and labels look correct.
      │
      ▼
4. Holdfast detection  (find_holdfast.py, called automatically by loader.py)
   For each cell, finds the holdfast — the foot where the cell attaches
   to the substrate. Results are cached as *_holdfasts.npy and reused
   on subsequent runs.
      │
      ▼
5. Load & preprocess  (loader.py)
   load_tiles() reads HDF5 tiles into (num_cells, H, W, total_frames).
   A 3×3 median filter is applied to each pre/post frame to suppress
   streak artifacts before the images are passed to the model.
   StentorPairs wraps tiles and labels into a PyTorch Dataset,
   pairing each pre/post frame and skipping NaN-labeled stimuli.
      │
      ▼
6. Holdfast masking  (loader.py → StentorPairs.__getitem__)
   Applies a circular mask (r=40 px) centered on the holdfast,
   zeroing out everything outside it so the model only sees the cell body.
      │
      ▼
7. Training  (train.py)
   Cells are split into disjoint train / val / test sets to prevent leakage.
   Each sample is a 2-channel image: [pre-stimulus frame, post-stimulus frame].
   The best checkpoint (by val F1) is saved to checkpoints/best_model.pt.
   After training, a threshold sweep (0.3–0.8) is printed and the F1-optimal
   threshold is selected automatically.
   Per-dataset failure .npz files are saved to outputs/ for diagnosis.
      │
      ▼
8. Failure analysis  (visualize_failures.py)
   Loads a failure .npz and saves grids of false positives and false negatives.
   Output PNG is auto-named to match the input .npz file.
      │
      ▼
9. Cross-dataset evaluation  (evaluate.py)
   Loads a trained checkpoint and evaluates it on a completely unseen recording
   (all cells used as test set, no training). Used to measure generalization.
```

---

## Model Architecture

```
Input: (batch, 2, 150, 150)        # pre and post frames stacked as channels

Conv(2→32,   k=3) + BN + ReLU + MaxPool(2)  →  (32,  75, 75)
Conv(32→64,  k=3) + BN + ReLU + MaxPool(2)  →  (64,  37, 37)
Conv(64→128, k=3) + BN + ReLU + MaxPool(2)  →  (128, 18, 18)
Conv(128→128,k=3) + BN + ReLU + MaxPool(2)  →  (128,  9,  9)
AdaptiveAvgPool(1) → Flatten → Dropout(0.3) →  (128,)
Linear(128 → 1)                              →  raw logit

Loss:      BCEWithLogitsLoss with pos_weight for class imbalance
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau on validation F1 (factor=0.5, patience=7)
```

---


## Installation

```bash
cd stentor_cnn
pip install -r requirements.txt
pip install scipy scikit-image
```

Python 3.9+ and PyTorch 2.2+ recommended.

---

## Usage

All scripts run from the `stentor_cnn/` directory.

### 1. Inspect a recording

```bash
python inspect_data.py ../tiles/RECORDING_tiled.h5 \
                       ../meta/RECORDING_tiled_data.h5 \
                       ../contraction/RECORDING_contractions.h5
```

Prints shapes, label distribution, and NaN counts. Always run this first on a new dataset.

### 2. Validate & smoke test

```bash
python data_integrity.py ../tiles/RECORDING_tiled.h5 \
                         ../meta/RECORDING_tiled_data.h5 \
                         ../contraction/RECORDING_contractions.h5

python smoke_test.py ../tiles/RECORDING_tiled.h5 \
                     ../meta/RECORDING_tiled_data.h5 \
                     ../contraction/RECORDING_contractions.h5
```

### 3. Train

```bash
python train.py ../tiles/RECORDING_tiled.h5 \
                ../meta/RECORDING_tiled_data.h5 \
                ../contraction/RECORDING_contractions.h5
```

To train on multiple recordings, pass additional triplets:

```bash
python train.py \
  ../tiles/REC1_tiled.h5 ../meta/REC1_tiled_data.h5 ../contraction/REC1_contractions.h5 \
  ../tiles/REC2_tiled.h5 ../meta/REC2_tiled_data.h5 ../contraction/REC2_contractions.h5
```

Key hyperparameters at the top of `train.py`:

```python
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
DROPOUT       = 0.3
SEED          = 42

VAL_CELLS  = None   # e.g. [16, 17, 18] to pin validation cells
TEST_CELLS = None   # e.g. [19, 20, 21] to pin test cells
```

Saves best checkpoint to `checkpoints/best_model.pt` and per-dataset failure `.npz` files to `outputs/`.

### 4. Visualize failure cases

```bash
python visualize_failures.py outputs/failures_RECORDING.npz
```

Saves a grid of false positives and false negatives to `outputs/`.

### 5. Evaluate on unseen data

```bash
python evaluate.py ../tiles/RECORDING_tiled.h5 \
                   ../meta/RECORDING_tiled_data.h5 \
                   ../contraction/RECORDING_contractions.h5 \
                   checkpoints/best_model.pt
```

Runs the trained model on a completely unseen recording with no retraining. All cells are used as the test set. Prints precision, recall, F1, and saves a failure `.npz`.

### 6. Export to ONNX

```bash
python export_onnx.py checkpoints/best_model.pt
```

---

## Upcoming: Cell-by-Cell Trajectory Approach

The current pipeline treats each stimulus independently — the model sees one pre/post pair in isolation with no memory of what came before. The next version will restructure the approach entirely:

Instead of looping over stimuli across all cells, the new pipeline loops over cells across all stimuli. For each cell, the model watches its entire stimulus history in order — all 120 stimuli from start to finish — and learns the temporal behavior pattern: when the cell was responding heavily, when it started habituating, and when it stopped contracting altogether. This gives the model full context for each prediction rather than treating every stimulus as if it were the first.

This approach is motivated by the strong habituation signal visible in the data: cells that contract frequently early in a recording contract far less by the end. A model that understands this trajectory makes fundamentally better predictions than one operating blind to temporal context.

---

## License

MIT — see [LICENSE](LICENSE).
