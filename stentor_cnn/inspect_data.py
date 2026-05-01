"""
-------
Print the shape, dtype, and label statistics of a Stentor recording. 
Re-run this on any new recording before trying to train on it 

Usage:
    python inspect_data.py <tiled.h5> <tiled_data.h5> <contractions.h5>

Example:  
    python inspect_data.py \\
        "../2025_09_12_01_33_36_tiled.h5" \\
        "../2025_09_12_01_33_36_tiled_data.h5" \\
        "../2025_09_12_01_33_36_contractions.h5"

"""
import sys
import h5py
import numpy as np 


def inspect_metadata(path: str) -> None:
    print(f"\n==== Metadata file: {path} ===")
    with h5py.File(path, "r") as f:
        for k in f.keys():
            d = f[k]
            if not isinstance(d, h5py.Dataset):
                continue 
            if d.shape == ():
                print(f"  {k:12s}  scalar           dtype={d.dtype}  value={d[()]}")
            else:
                print(f"  {k:12s}  shape={str(d.shape):14s}  dtype={d.dtype}")

def inspect_tiled(path:str)-> None:
    print(f"\n===Tiled file: {path}===") 
    with h5py.File(path, "r") as f:
        trial_keys = sorted(k for k in f.keys() if k.startswith("tiled_frames_trial_")) 
        print(f" {len(trial_keys)} trial(s): {trial_keys}")
        for k in trial_keys:
            d = f[k]
            arr = d[:]
            print(f" {k}: shape={d.shape}  dtype={d.dtype}   "
                  f"min={arr.min()} max={arr.max()}  mean={arr.mean():.2f}")

def inspect_contractions(path: str) -> None:
    print(f"\n=== Contractions file: {path} ===")
    with h5py.File(path, "r") as f:
        for k in f.keys():
            arr = f[k][:]
            n_one = int(np.sum(arr == 1))
            n_zero = int(np.sum(arr == 0))
            n_nan = int(np.sum(np.isnan(arr)))
            total = arr.size
            print(f"  {k:8s}  shape={str(arr.shape):10s}  dtype={arr.dtype}  "
                  f"ones={n_one}  zeros={n_zero}  nans={n_nan}  total={total}")
            if k == "manual" and total>0:
                pos_frac = n_one / max(total - n_nan, 1)
                print(f"            positive fraction: {pos_frac:.1%}")

def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1 

    tiled, meta, gt = sys.argv[1], sys.argv[2], sys.argv[3]
    inspect_metadata(meta)
    inspect_tiled(tiled)
    inspect_contractions(gt)
    print("\n inspection complete!\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())