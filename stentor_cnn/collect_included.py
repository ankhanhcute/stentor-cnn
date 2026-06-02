"""
collect_included.py
-------------------
Finds all recordings whose manual contraction labels pass the >= 50% threshold
(same rule as check_data.py) and copies the matching contraction, tiled, and
meta h5 files into one output folder: ../included_data/

Usage (from project root or stentor_cnn/):
    python stentor_cnn/collect_included.py
"""

import glob
import os
import shutil
import h5py
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

CONTRACTION_DIRS = [
    os.path.join(PROJECT_ROOT, "contraction"),
    os.path.join(PROJECT_ROOT, "send_Khanh"),
]

TILED_DIRS = [
    os.path.join(PROJECT_ROOT, "send_Khanh"),
    os.path.join(PROJECT_ROOT, "contraction"),
]

META_DIRS = [
    os.path.join(PROJECT_ROOT, "meta"),
    os.path.join(PROJECT_ROOT, "send_Khanh"),
    os.path.join(PROJECT_ROOT, "contraction"),
]

OUT_DIR = os.path.join(PROJECT_ROOT, "included_data")
THRESHOLD = 50.0  # minimum % of ones in manual labels to include


def find_file(stem: str, suffix: str, search_dirs: list[str]) -> str | None:
    """Return the first path matching <stem><suffix> found in search_dirs."""
    for d in search_dirs:
        candidate = os.path.join(d, stem + suffix)
        if os.path.isfile(candidate):
            return candidate
    return None


def is_included(contraction_path: str) -> tuple[bool, float]:
    with h5py.File(contraction_path, "r") as f:
        if "manual" not in f:
            return False, 0.0
        m = f["manual"][()]
        ones = np.nansum(m == 1)
        total = np.sum(~np.isnan(m))
        rate = ones / total * 100 if total > 0 else 0.0
    return rate >= THRESHOLD, rate


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    contraction_paths = sorted(
        p for d in CONTRACTION_DIRS for p in glob.glob(os.path.join(d, "*_contractions.h5"))
    )

    included: list[str] = []
    excluded: list[str] = []

    print(f"{'status':<9}  {'rate':>6}  file")
    print("-" * 70)

    for c_path in contraction_paths:
        ok, rate = is_included(c_path)
        status = "INCLUDE" if ok else "EXCLUDE"
        print(f"{status:<9}  {rate:5.1f}%  {c_path}")
        if ok:
            included.append(c_path)

    print(f"\n{len(included)} included / {len(contraction_paths)} total")
    print(f"Copying files → {OUT_DIR}\n")

    copied = 0
    missing = []

    for c_path in included:
        basename = os.path.basename(c_path)                      # e.g. 2024_10_12_02_53_41_contractions.h5
        stem = basename.replace("_contractions.h5", "")          # e.g. 2024_10_12_02_53_41

        files_to_copy = {
            "contractions": (c_path, f"{stem}_contractions.h5"),
            "tiled":        (find_file(stem, "_tiled.h5", TILED_DIRS),      f"{stem}_tiled.h5"),
            "meta":         (find_file(stem, "_tiled_data.h5", META_DIRS),   f"{stem}_tiled_data.h5"),
        }

        for kind, (src, dest_name) in files_to_copy.items():
            if src is None:
                print(f"  WARNING: no {kind} file found for {stem}")
                missing.append(f"{stem} ({kind})")
                continue
            dest = os.path.join(OUT_DIR, dest_name)
            shutil.copy2(src, dest)
            print(f"  copied {kind:<12}  {dest_name}")
            copied += 1

    print(f"\nDone. {copied} files copied to {OUT_DIR}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
