import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stentor_cnn'))
import numpy as np
from skimage.filters import median
import loader
from find_holdfast import find_holdfast

DATA = '/n/home06/ktruong/projects/stentor-cnn/data'

HOLDOUT = {
    '2025_10_20_03_02_39', '2025_11_02_23_30_22', '2025_11_03_20_45_41',
    '2025_10_30_20_47_23', '2025_11_01_17_58_50', '2025_09_22_02_43_12',
    '2025_04_26_23_21_10', '2025_06_12_22_58_40', '2025_04_14_20_26_34',
    '2025_05_25_00_54_10', '2025_10_29_23_54_44', '2025_10_30_20_46_05',
    '2025_11_02_23_29_03', '2025_11_02_23_31_42', '2025_09_22_02_41_52',
    '2025_09_23_19_20_47', '2025_06_13_23_05_12', '2025_05_30_20_11_44',
    '2025_04_26_23_24_08', '2024_12_29_03_27_19', '2024_11_10_11_53_54',
    '2026_05_05_17_39_45',
}

names = []
for f in sorted(os.listdir(DATA)):
    if f.endswith('_tiled.h5'):
        name = f.replace('_tiled.h5', '')
        if name in HOLDOUT:
            continue
        t = os.path.join(DATA, f"{name}_tiled.h5")
        m = os.path.join(DATA, f"{name}_tiled_data.h5")
        c = os.path.join(DATA, f"{name}_contractions.h5")
        if os.path.exists(t) and os.path.exists(m) and os.path.exists(c):
            names.append(name)

print(f"Building cache for {len(names)} datasets...")
for i, name in enumerate(names):
    tiled_path = os.path.join(DATA, f"{name}_tiled.h5")
    processed_cache = tiled_path.replace('.h5', '_processed.npy')
    if os.path.exists(processed_cache):
        print(f"  [{i+1}/{len(names)}] already cached: {name}")
        continue
    print(f"  [{i+1}/{len(names)}] processing: {name}")
    meta_path = os.path.join(DATA, f"{name}_tiled_data.h5")
    tiles, meta = loader.load_tiles(tiled_path, meta_path)
    crop_size = tiles.shape[1]
    holdfasts_cache = tiled_path.replace('.h5', '_holdfasts.npy')
    if os.path.exists(holdfasts_cache):
        holdfasts = list(np.load(holdfasts_cache))
    else:
        holdfasts = []
        for c in range(tiles.shape[0]):
            result = find_holdfast(tiles[c], crop_size)
            holdfasts.append(result['holdfast'])
        np.save(holdfasts_cache, np.array(holdfasts))
    processed = np.empty_like(tiles)
    for c in range(tiles.shape[0]):
        cy, cx = holdfasts[c]
        h, w = tiles.shape[1], tiles.shape[2]
        mask = loader.make_circular_mask(h, w, cy, cx, r=40)
        for fr in range(tiles.shape[3]):
            frame = tiles[c, :, :, fr] * mask
            processed[c, :, :, fr] = median(frame, footprint=np.ones((3, 3)))
    tmp = processed_cache.replace("_processed.npy", "_processed_tmp.npy")
    np.save(tmp, processed)
    os.replace(tmp, processed_cache)
    print(f"    saved → {processed_cache}")

print("All caches built!")
