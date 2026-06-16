# Stentor CNN — Automated Contraction Detection

A deep learning pipeline for automated detection of *Stentor coeruleus* contractions from microscopy recordings. Replaces manual annotation with a CNN-based classifier that outputs per-stimulus contraction predictions.

---

## Overview

Stentor coeruleus cells contract in response to mechanical stimuli. This pipeline takes raw microscopy recordings, preprocesses them, and outputs a prediction (contracted / not contracted / uncertain) for each stimulus across all cells in a recording.

**Model:** `StentorSequenceModel` — a CNN backbone with a temporal Conv1d layer that processes sequences of pre/post stimulus frame pairs per cell.

**Performance on unseen datasets:**
- Mean F1: 0.952
- Mean Recall: 0.952

---

## Setup

```bash
conda activate stentor
```

All scripts run from the project root (`stentor_cnn/`).

---

## Pipeline

### 1. Preprocess (build disk cache)

Must be run before training. Computes holdfast positions and applies median filtering, saving results to disk so training doesn't recompute every epoch.

```bash
sbatch preprocess.sbatch
```

Or manually:
```bash
python build_cache.py <tiled.h5> <meta.h5>
```

---

### 2. Train

```bash
python train.py <tiled.h5> <meta.h5> <contractions.h5>
```

Multiple datasets:
```bash
python train.py \
    tiles/2024_11_07_tiled.h5 meta/2024_11_07_tiled_data.h5 contraction/2024_11_07_contractions.h5 \
    tiles/2024_11_09_tiled.h5 meta/2024_11_09_tiled_data.h5 contraction/2024_11_09_contractions.h5
```

On the cluster:
```bash
sbatch train_stentor.sbatch
```

**Outputs:**
- `checkpoints/best_model.pt` — best model checkpoint
- `checkpoints/best_thresh.json` — optimal decision threshold
- `outputs/training_curves.png` — loss/accuracy/F1 curves

---

### 3. Evaluate (on labeled data)

Run on datasets that have ground truth labels to measure model performance.

```bash
python evaluate.py <tiled.h5> <meta.h5> <contractions.h5> [checkpoint.pt]
```

Example:
```bash
python evaluate.py \
    tiles/2024_12_06_tiled.h5 \
    meta/2024_12_06_tiled_data.h5 \
    contraction/2024_12_06_contractions.h5
```

**Outputs:**
- Console: loss, precision, recall, F1, threshold sweep
- `outputs/predictions_{dataset}.json` — per-stimulus predictions
- `outputs/uncertain_{dataset}.json` — uncertain stimuli detail
- `outputs/failures_{dataset}.npz` — FP/FN tiles for visualization

---

### 4. Predict (on new unlabeled data)

Run on new recordings with no ground truth labels.

```bash
python predict.py <tiled.h5> <meta.h5> [checkpoint.pt]
```

Example:
```bash
python predict.py \
    tiles/2025_01_10_tiled.h5 \
    meta/2025_01_10_tiled_data.h5
```

**Outputs:**
- `outputs/predictions_{dataset}.json` — per-stimulus predictions
- `outputs/uncertain_{dataset}.json` — uncertain stimuli

---

## Output Format

### predictions_{dataset}.json

A list of per-stimulus predictions:

```json
[
  {"cell": 0, "stimulus": 5, "prediction": 1, "probability": 0.9937},
  {"cell": 0, "stimulus": 6, "prediction": 0, "probability": 0.0003},
  {"cell": 1, "stimulus": 3, "prediction": null, "probability": 0.78}
]
```

| prediction | meaning |
|---|---|
| `1` | contracted |
| `0` | not contracted |
| `null` | uncertain — model not confident enough |

**Uncertainty logic (asymmetric):**
- If `prob >= threshold`: uncertain when `abs(prob - threshold) < 0.05`
- If `prob < threshold`: uncertain when `abs(prob - threshold) < 0.1`
- Human-labeled NaN → always null

The asymmetry is intentional — we're stricter about flagging uncertain contractions than uncertain non-contractions, to protect recall.

### uncertain_{dataset}.json

Full detail on every uncertain stimulus:

```json
[
  {"cell": 1, "stimulus": 3, "probability": 0.78}
]
```

---

## File Overview

| File | Purpose |
|---|---|
| `model.py` | Model architecture (`StentorSequenceModel`) |
| `loader.py` | Data loading, preprocessing, `StentorPairs` dataset |
| `find_holdfast.py` | Locates the holdfast anchor point per cell |
| `train.py` | Training loop |
| `evaluate.py` | Evaluation on labeled data |
| `predict.py` | Inference on unlabeled data |
| `build_cache.py` | Precomputes processed frames to disk |
| `build_holdout_cache.py` | Cache builder for holdout datasets |
| `visualize_failures.py` | Visualizes FP/FN cases as PNGs |
| `cross_val/` | 5-fold cross-validation scripts |

---

## Known Issues / Excluded Datasets

- `2024_11_10` — too noisy, excluded from training
- `2026_05_05` — near-zero contraction rate, excluded from training
- Cells with ≥75% NaN labels are automatically skipped during training
- Artifact cells (streak artifacts) reduce precision but not recall — all failures trace to artifact cells

---

## Notes

- The 3×3 median filter in `loader.py` meaningfully reduces streak artifact noise — do not remove it
- Circular mask (r=40px) around holdfast tip is used because contraction occurs at the holdfast specifically
- `best_thresh.json` must exist before running `predict.py` — it is generated by `train.py`
