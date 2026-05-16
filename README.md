# Stentor CNN

A compact convolutional neural network for detecting contraction events in *Stentor coeruleus* (a single-celled ciliate) from paired pre- and post-stimulus microscopy frames.

## Overview

*Stentor coeruleus* contracts in response to mechanical or chemical stimuli. This project automates contraction detection by training a binary classifier on paired image tiles: one frame captured just before a stimulus and one immediately after. The model predicts whether the cell contracted.

Key properties:
- **~275k parameter CNN** — fast to train, easy to inspect
- **Cell-disjoint data splits** — train/val/test are split by cell identity to prevent data leakage
- **Holdfast masking** — automatically detects and isolates the cell's attachment point (foot), zeroing out background pixels so the model focuses on relevant anatomy
- **Multi-recording training** — accepts any number of recordings concatenated at the command line

## Repository Structure

```
.
├── stentor_cnn/
│   ├── model.py              # StentorCNN architecture
│   ├── loader.py             # HDF5 data loading, Dataset class, holdfast masking
│   ├── train.py              # Training loop with cell-disjoint splits
│   ├── find_holdfast.py      # Holdfast detection (segmentation + skeleton fallback)
│   ├── smoke_test.py         # Visual sanity check of data loading
│   ├── inspect_data.py       # Print data statistics (shapes, label distribution)
│   ├── data_integrity.py     # Automated data validation checks
│   ├── visualize_failures.py # Plot false positives and false negatives
│   ├── export_onnx.py        # (Placeholder) ONNX export
│   ├── requirements.txt      # Python dependencies
│   ├── checkpoints/          # Saved model weights (best_model.pt)
│   └── outputs/              # Training curves and debug visualizations
├── tiles/                    # Pre-processed cell tile HDF5 files (*_tiled.h5)
├── meta/                     # Metadata HDF5 files (*_tiled_data.h5)
├── contraction/              # Ground-truth label HDF5 files (*_contractions.h5)
└── auto_annotate_v3.jl       # Julia preprocessing pipeline (raw video → HDF5 tiles)
```

## Data Format

Each recording produces three HDF5 files:

| File | Contents |
|------|----------|
| `*_tiled.h5` | Cell tile images, one dataset per trial (`tiled_frames_trial_0`, …) |
| `*_tiled_data.h5` | Metadata scalars: `crop_size`, `num_cells`, `row_num`, `col_num` |
| `*_contractions.h5` | Manual labels, dataset `manual`, shape `(num_cells, num_stims)`, values 0/1/NaN |

**Frame convention**: for stimulus `k`, the pre-stimulus frame is at index `2k` and the post-stimulus frame is at `2k+1`. The loader stacks these into a 2-channel input of shape `(2, 150, 150)`.

NaN labels are excluded from training and evaluation.

## Pipeline Overview

The full workflow goes from raw video to a trained classifier in six stages:

```
Raw video (.mp4)
      │
      ▼
1. Preprocessing  (auto_annotate_v3.jl)
   Detects individual Stentor cells, crops each one into a 150×150 tile,
   and saves paired frames (one before each stimulus, one after) into HDF5.
   Produces: *_tiled.h5, *_tiled_data.h5, *_contractions.h5
      │
      ▼
2. Inspect & validate  (inspect_data.py, data_integrity.py)
   inspect_data.py prints shapes, label distribution, and NaN counts.
   data_integrity.py runs automated checks on tile normalization, frame
   pairing, and label format. Run these before training to catch bad data.
      │
      ▼
3. Smoke test  (smoke_test.py)
   Saves a visual grid of pre/post frame pairs to outputs/ so you can
   manually verify the tiles and labels look correct.
      │
      ▼
4. Holdfast detection  (find_holdfast.py, runs automatically inside loader.py)
   For each cell tile, finds the holdfast — the foot where the cell attaches
   to the substrate. Applies a circular mask (r=40 px) centered on the
   holdfast so the model only sees the cell body, not background.
   Produces: *_holdfasts.npy  (cached, reused on subsequent runs)
      │
      ▼
5. Training  (train.py)
   Cells are shuffled and split into disjoint train / val / test sets so
   the model never sees the same cell in both training and evaluation.
   Each sample is a 2-channel image: [pre-stimulus frame, post-stimulus frame].
   The CNN outputs a contraction probability; the best checkpoint (by val F1)
   is saved automatically. A threshold sweep is printed at the end.
   Produces: checkpoints/best_model.pt, outputs/training_curves.png
      │
      ▼
6. Failure analysis  (visualize_failures.py)
   Loads the saved checkpoint and runs it on the held-out test cells.
   Saves grids of false positives and false negatives so you can see
   which stimulus-response pairs the model got wrong.
   Produces: outputs/visualize_failure*.png
```

## Model Architecture

```
Input: (batch, 2, 150, 150)   # pre and post frames stacked as channels

Conv(2→32, k=3)  + BN + ReLU + MaxPool(2)   →  (32,  75, 75)
Conv(32→64, k=3) + BN + ReLU + MaxPool(2)   →  (64,  37, 37)
Conv(64→128,k=3) + BN + ReLU + MaxPool(2)   →  (128, 18, 18)
Conv(128→128,k=3)+ BN + ReLU + MaxPool(2)   →  (128,  9,  9)
AdaptiveAvgPool(1) → Flatten → Dropout(0.3) →  (128,)
Linear(128 → 1)                              →  raw logit
```

- **Loss**: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau on validation F1 (factor=0.5, patience=7)

## Installation

```bash
cd stentor_cnn
pip install -r requirements.txt
# also required (not yet in requirements.txt):
pip install scipy scikit-image
```

Python 3.9+ and PyTorch 2.2+ recommended.

## Usage

All scripts run from the `stentor_cnn/` directory and accept paths to the three HDF5 files for each recording.

### 1. Inspect data

```bash
python inspect_data.py ../tiles/RECORDING_tiled.h5 \
                       ../meta/RECORDING_tiled_data.h5 \
                       ../contraction/RECORDING_contractions.h5
```

Prints shapes, dtypes, label distribution, and NaN counts.

### 2. Validate data integrity

```bash
python data_integrity.py ../tiles/RECORDING_tiled.h5 \
                         ../meta/RECORDING_tiled_data.h5 \
                         ../contraction/RECORDING_contractions.h5
```

Runs automated checks on tile normalization, frame pairing, and label format.

### 3. Visual smoke test

```bash
python smoke_test.py ../tiles/RECORDING_tiled.h5 \
                     ../meta/RECORDING_tiled_data.h5 \
                     ../contraction/RECORDING_contractions.h5
```

Saves a grid of pre/post frame pairs to `outputs/` so you can visually verify the data looks correct before training.

### 4. Train

```bash
python train.py ../tiles/RECORDING_tiled.h5 \
                ../meta/RECORDING_tiled_data.h5 \
                ../contraction/RECORDING_contractions.h5
```

To train on multiple recordings, pass additional triplets:

```bash
python train.py ../tiles/REC1_tiled.h5 ../meta/REC1_tiled_data.h5 ../contraction/REC1_contractions.h5 \
                ../tiles/REC2_tiled.h5 ../meta/REC2_tiled_data.h5 ../contraction/REC2_contractions.h5
```

Hyperparameters are set as constants at the top of `train.py`:

```python
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
DROPOUT       = 0.3
SEED          = 42

VAL_CELLS  = None   # set to e.g. [16, 17, 18] to pin validation cells
TEST_CELLS = None   # set to e.g. [19, 20, 21] to pin test cells
```

Training saves the best checkpoint (by validation F1) to `checkpoints/best_model.pt` and writes loss/accuracy/F1 curves to `outputs/training_curves.png`. A threshold sweep on the test set (0.3–0.8) is printed at the end.

### 5. Visualize failure cases

```bash
python visualize_failures.py ../tiles/RECORDING_tiled.h5 \
                             ../meta/RECORDING_tiled_data.h5 \
                             ../contraction/RECORDING_contractions.h5 \
                             checkpoints/best_model.pt
```

Saves grids of false positives and false negatives to `outputs/`.

### 6. Detect holdfasts (optional, runs automatically during training)

```bash
python find_holdfast.py ../tiles/RECORDING_tiled.h5 \
                        ../meta/RECORDING_tiled_data.h5
```

Detects each cell's attachment point and caches results to `../tiles/RECORDING_holdfasts.npy`. The loader uses this cache automatically on subsequent runs.

Results vary by recording and cell-split randomness; run the threshold sweep printed after training to tune precision/recall for your use case.

## License

MIT — see [LICENSE](LICENSE).
