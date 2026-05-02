"""
data_integrity.py
----------------
This script is sanity check after loading all the data and smoke test. To 
make sure all the data for tiles, labels and meta in the right format 
     1. Tiles :
     - 4D 
     - dtype: float32 (CNN just accept float32)
     - min is >= 0 and max is <= 1(normalized)
     - No NaN or Inf values in pixel data 
      - H and W both 150 
      - total_frames is even ( because every stimulus need pre and post)
      2. Labels:
      - 2D
      - All values 1.0, 0,0, NaN 
      - NaN count is reasonable 
      - Not all the labels are the same value
      - Positive fraction between 5% and 95%
      3. Pairing:
      - tiles.shape[0] == number of cells 
      - tiles.shape[3] == manual.shape * 2 = total stimuli * 2
      - total_frames == num_trails * num_stims * 2
      Run from the project root:
        python stentor_cnn/data_integrity.py <tiled.h5> <meta.h5> <contractions.h5>

       Output:
         Prints a PASS/WARN report to the terminal for each check.
         No files saved.
"""

import sys
import os 
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, THIS_DIR)
import loader

#------paths------
if len(sys.argv) != 4:
    print("Usage: python data_integrity.py <tiled.h5> <meta.h5> <contractions.h5>")
    sys.exit(1)

TILED_H5 = sys.argv[1]
META_H5 = sys.argv[2]
GT_H5 = sys.argv[3]

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
    

    print("====Tiles check====")

    if tiles.ndim == 4:
        print(f"[PASS] tiles is 4D: {tiles.shape}")
    else:
        print(f"[FAILED] tiles is not 4D: {tiles.shape}")
    
    if tiles.dtype == np.float32:
        print("[PASS] dtype is float32")
    else:
        print(f"[WARN] dtype is {tiles.dtype}, expected to be float32")
    if tiles.min() >= 0.0 and tiles.max() <= 1.0:
        print(f"[PASS] tiles normalized: min={tiles.min():.3f} max={tiles.max():.3f}")
    else:
        print(f"[WARN] tiles not normalized: min={tiles.min():.3f} max={tiles.max():.3f}")
    if not np.any(np.isnan(tiles)) and not np.any(np.isinf(tiles)):
        print("[PASS] no NaN or Inf in tiles")
    else:
        print("[WARN] NaN or Inf found in tiles!")
    if tiles.shape[1] ==  150 and tiles.shape[2] == 150:
        print(f"[PASS] H and W are both 150")
    else:
        print(f"[WARN] unexpected H or W: H={tiles.shape[1]} w={tiles.shape[2]}")
    total_frames = tiles.shape[3]
    if total_frames % 2 == 0:
        print(f"[PASS] total_frames is even: {total_frames}")
    else:
        print(f"[WARN] total_frames is odd: {total_frames}, every stimulus needs pre AND post") 
    
    print("====Labels check===")
    if manual.ndim == 2:
        print(f"[PASS] labels is 2D: {manual.shape}")
    else:
        print(f"[FAILED] {manual.shape} is not 2D, expected 2D")
    valid = np.all((manual == 0.0) | (manual == 1.0) | np.isnan(manual))
    if valid:
        print("[PASS] all label values are 0.0, 1.0, or NaN")
    else:
        print("[WARN] unexpected values found in labels")
    nan_count = int(np.sum(np.isnan(manual)))
    nan_frac = nan_count / manual.size
    if nan_frac < 0.1:
        print(f"[PASS] NaN count is reasonable: {nan_count}/{manual.size} ({nan_frac:.1%})")
    else:
        print(f"[WARN] too many NaNs: {nan_count}/{manual.size} ({nan_frac:.1%})")
    pos_frac = np.nanmean(manual)
    if 0.05 <= pos_frac <= 0.95:
        print(f"[PASS] positive fraction is reasonable: {pos_frac:.1%}")
    else:
        print(f"[WARN] positive fraction is very skewed: {pos_frac:.1%}")
    unique_val = np.unique(manual[~np.isnan(manual)])
    if len(unique_val) > 1:
        print(f"[PASS] labels have both 0 and 1 values")
    else:
        print(f"[WARN] all labels are the same value: {unique_val}")

    print("====Pairing check====")
    if tiles.shape[0] == manual.shape[0]:
        print(f"[PASS] Number of cells on both {tiles.shape[0]} and {manual.shape[0]} are equal ")
    else:
        print(f"[WARN] Number of cells in {tiles.shape[0]} and {manual.shape[0]} are not equal, expected to be ")
    if tiles.shape[3] == manual.shape[1] * 2:
        print(f"[PASS] we have correct number of frames: {tiles.shape[3]}")
    else:
        print(f"[WARN] we didnt have the correct number of frames: {tiles.shape[3]}")
    if meta['total_frames'] == meta['num_trials'] * meta['num_stims_per_trial'] * 2:
        print(f"[PASS] total frames: {meta['total_frames']} is equal to the num_trials:{meta['num_trials']} x num_stims: {meta['num_stims_per_trial']} x 2")
    else:
        print(f"[WARN] total frames: {meta['total_frames']} are not correct")
    
    print("\n=== Integrity check complete ===")
    return 0






if __name__ == "__main__":
    sys.exit(main())