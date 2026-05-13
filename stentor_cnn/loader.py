"""
loader.py

-------
Data loading utilites for the Stentor coeruleus CNN pipeline.

Three public entry points:
    load_tiles(tiled_h5, meta_h5)        -> (tiles, meta_dict)
    load_manual_labels(contractions_h5)  -> ndarray (num_cells, total_stim)
    StentorPairs(...)                    -> torch.utils.data.Dataset

Frame-pairing convention (0-indexed):
      For stimulus k in [0, total_stim]:
      pre frame index = 2*k
      post frame index = 2*k + 1

"""

from __future__ import annotations

from typing import Iterable, Optional
import os
import h5py
import numpy as np
import torch 
from torch.utils.data import Dataset 
#-----low level loading-----
def load_tiles(tiled_h5_path:str, meta_h5_path:str)  -> tuple[np.array, dict]:
    """
    Load all trials from a Stentor tiled HD5F file and reshape it 
    into per-cell tiles.
    Parameters
    ----------
    tiled_h5_path : path to "{stamp}_tiled.h5".
    meta_h5_path  : path to "{stamp}_tiled_data.h5".
    Returns
    ----------
    tiles : np.ndarray, shape (num_cells, H, W, total_frames), float32 in [0, 1].
    meta  : dict with keys
        crop_size, num_cells, num_trials, num_stim_per_trial, total_stim,
        total_frames.
    """

    with h5py.File(meta_h5_path, "r") as m:
        crop_size = int(m["crop_size"][()])
        num_cells = int(m["num_cells"][()])

    with h5py.File(tiled_h5_path, "r") as f:
        trial_keys = sorted(
        k for k in f.keys() if k.startswith("tiled_frames_trial_")
            )
        if not trial_keys:
            raise ValueError(
            f" No 'tiled_frames_trial_*' dataset found in {tiled_h5_path}"
        )    

        per_trials: list[np.ndarray] = []
        for k in trial_keys:
            raw = f[k][:]

            if raw.ndim !=3:
                raise ValueError(
                f" {k} has shape {raw.shape}; expected 3 dims"
            )
        
            T, H, big = raw.shape
            if H != crop_size or big != num_cells * crop_size:
                raise ValueError(
                    f"{k} shape {raw.shape} inconsistent with metadata "
                    f"crop_size={crop_size}, num_cells={num_cells}"
                )
            x = raw.reshape(T, crop_size, num_cells, crop_size)
            tiled = np.transpose(x, (2, 3, 1, 0))
            per_trials.append(tiled)

        tiles_u8 = np.concatenate(per_trials, axis=-1)
        tiles = tiles_u8.astype(np.float32) / 255.0

        num_trials = len(trial_keys)
        total_frames = tiles.shape[-1]
        if total_frames % (2 * num_trials) != 0:
            raise ValueError(
            f"total_frames {total_frames} not divisible by 2*num_trials "
            f"({2*num_trials}); expected even number of frames per trial."
        )

        num_stims_per_trial = total_frames // (2*num_trials)
        total_stims = num_stims_per_trial * num_trials

        meta = dict(
        crop_size=crop_size, 
        num_cells=num_cells,
        num_trials=num_trials,
        num_stims_per_trial=num_stims_per_trial,
        total_stims=total_stims, 
        total_frames=total_frames,
    )

    return tiles, meta

def load_manual_labels(contractions_h5_path: str) -> np.ndarray:
    """
    Read the human-annotated 'manual' array from the contraction h5 file

    Returns
    -----
    manual : ndarray, shape (num_cells, total_stim), dtype float64.
        Values in {0.0, 1.0, NaN}. NaN means "skip this sample".
    """

    with h5py.File(contractions_h5_path, "r") as f:
        if "manual" not in f:
            raise KeyError(
            f"'manual' dataset not found in {contractions_h5_path}; "
                f"keys present: {list(f.keys())}"
        )
        return f["manual"][:]
def make_circular_mask(h, w, cy, cx, r):
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
        return (ys - cy)**2 + (xs - cx)**2 <= r**2
#-----PyTorch Dataset------ß

class StentorPairs(Dataset):
    """
    One sample = (pre_frame, post_frame) stacked as 2 channels, plus a binary 
    label.

    Parameters
    ---------
    tiles: np.ndrray, shape (num_cells, H, W, total_frame), float32 in [0,1].
       From load_tiles().
    manual : np.ndarray, shape (num_cells, total_frame), float64.
      From load_maul_labels().NaN entries are skipped.
    cell_indices : iterable of int, the cells to include in this Dataset.
      Pass disjoint subsets to build train/val/test sets.
    """
    def __init__(
        self,
        tiles: np.ndarray,
        manual: np.ndarray,
        cell_indices: Iterable[int], #to avoid training and testing on the same cells
        tiled_h5_path: str = "",
    ):  
        
        if tiles.ndim != 4: #cell, height, width, frame
            raise ValueError(f"tiles must be 4-D, got shape {tiles.shape}")
        if manual.ndim != 2: #cell x stimulus
            raise ValueError(f"manual must be 2-D, got shape {manual.shape}")
        num_cells_t = tiles.shape[0]
        num_cells_m, total_stim = manual.shape
        if num_cells_t != num_cells_m:
            raise ValueError(
                f"num_cells mismatch: tiles has {num_cells_t}, manual has {num_cells_m}"
            )
        if tiles.shape[-1] != 2 * total_stim: #Do we have exactly 2 image frames for every manual label?
            raise ValueError(
                f"frame/label mismatch: tiles has {tiles.shape[-1]} frames "
                f"but manual implies {2*total_stim}"
            )
        #saving data inside class
        self.tiles = tiles
        self.manual = manual
        from find_holdfast import find_holdfast
        crop_size = tiles.shape[1]
        self.holdfasts = []
        cache_path = tiled_h5_path.replace(".h5", "_holdfasts.npy")
        
        if os.path.exists(cache_path):
    # already computed before, just load it
            self.holdfasts = list(np.load(cache_path))
        else:
    # first time, compute and save
            for c in range(tiles.shape[0]):
                result = find_holdfast(tiles[c], crop_size)
                self.holdfasts.append(result['holdfast'])
    np.save(cache_path, np.array(self.holdfasts))
        self.index: list[tuple[int, int, float]] = []
        cell_indices = list(cell_indices)
        for c in cell_indices:
            if not (0 <= c < num_cells_t):
                raise IndexError(f"cell index {c} out of range [0, {num_cells_t})")
            for k in range(total_stim):
                lab = manual[c, k]
                if np.isnan(lab):
                    continue
                self.index.append((int(c), int(k), float(lab))) #cell, stimulus, label
    def __len__(self) -> int: #tell pytorch how many usable sample we have
        return len(self.index)
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        c, k, lab = self.index[i]
        pre = self.tiles[c, :, :, 2 * k]
        post = self.tiles[c, :, :, 2 * k + 1]
        cy, cx = self.holdfasts[c]
        h, w = pre.shape
        mask = make_circular_mask(h, w, cy, cx, r=40)
        pre = pre * mask
        post = post * mask
        x = np.stack([pre, post], axis=0)         # (2, H, W)

        return(
            torch.from_numpy(x),
            torch.tensor(lab, dtype=torch.float32),)
    def positive_fraction(self) -> float:
        if not self.index:
            return 0.0
        return float(np.mean([lab for _, _, lab in self.index]))

#-----build the train/val/test by cell-disjoint split

def make_cell_disjoint_split(
    num_cells: int, 
    val_cells: Optional[Iterable[int]] = None, 
    test_cells: Optional[Iterable[int]] = None, 
    seed : int = 0,
    val_frac: float = 0.15, 
    test_frac: float = 0.15,

) -> tuple[list[int], list[int], list[int]]:
    """
    Partition cells into disjoint train/val/test sets.
    If val_cells/test_cells are given explicitly, those are used. Otherwise
    cells are shuffled (seeded) and split by fraction.
    """
    all_cells = list(range(num_cells))

    if val_cells is not None or test_cells is not None:
        val_cells = list(val_cells or [])
        test_cells = list(test_cells or [])
        forbidden = set(val_cells) | set(test_cells)
        train = [c for c in all_cells if c not in forbidden]
        return train, val_cells, test_cells

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(num_cells))
    n_test = max(1, int(round(num_cells * test_frac)))
    n_val = max(1, (round(num_cells * val_frac)))
    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test:n_test + n_val])
    train = sorted(shuffled[n_test + n_val:])
    return train, val, test