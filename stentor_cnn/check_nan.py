
import argparse
import json
import numpy as np
import os
import h5py
from find_holdfast import segment_frame_for_holdfast, detect_static_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="to easier import cached build and uncertainty cases json")
    parser.add_argument("prediction_json", help="path to predictions_{dataset}.json")
    parser.add_argument("processed_npy", help="path to _processed_{dataset}.npy")
    parser.add_argument("tiled_h5", help="path to {dataset}_tiled.h5")
    parser.add_argument("meta_h5", help="path to {dataset}_tiled_data.h5")
    return parser.parse_args()


def is_pre_contracted(pre_frame, bg_mean, bg_std, artifact_mask, center, crop_size):
    mask, score = segment_frame_for_holdfast(pre_frame, bg_mean, bg_std, artifact_mask, center, crop_size)
    area = np.sum(mask)
    if area < 20 or area > 80:
        return False
    ys, xs = np.where(mask)
    y_span = ys.max() - ys.min() + 1
    x_span = xs.max() - xs.min() + 1
    compactness = area / (y_span * x_span)
    cy_m, cx_m = np.mean(ys), np.mean(xs)
    dist = np.sqrt((cy_m - center[0])**2 + (cx_m - center[1])**2)
    if dist > crop_size * 0.3:
        return False
    return compactness >= 0.3 and score >= 10.0


def main():
    args = parse_args()
    with open(args.prediction_json, 'r') as file:
        predictions = json.load(file)
    processed = np.load(args.processed_npy, mmap_mode='r')
    with h5py.File(args.meta_h5, 'r') as m:
        crop_size = int(m['crop_size'][()])
        num_cells = int(m['num_cells'][()])
    with h5py.File(args.tiled_h5, 'r') as f:
        trial_keys = sorted(k for k in f.keys() if k.startswith('tiled_frames_trial_'))
        raw = np.concatenate([f[k][:] for k in trial_keys], axis=0)
    T, H, big = raw.shape
    raw = raw.reshape(T, H, num_cells, crop_size)
    raw = np.transpose(raw, (2, 1, 3, 0)).astype(np.float32) / 255.0
    pre_contracted_flags = []
    n_flagged = 0
    for entry in predictions:
        if entry["prediction"] is None:
            c = entry["cell"]
            k = entry["stimulus"]
            pre_frame = processed[c, :, :, 2 * k]
            tiles_cell = raw[c]  # (H, W, total_frames)
            bg_mean = np.mean(tiles_cell)
            bg_std = np.std(tiles_cell)
            artifact_mask = detect_static_artifacts(tiles_cell, bg_mean)
            center = (crop_size / 2.0, crop_size / 2.0)
            if is_pre_contracted(pre_frame, bg_mean, bg_std, artifact_mask, center, crop_size):
                entry["pre_contracted"] = True
                n_flagged += 1
                pre_contracted_flags.append({"cell": c, "stimulus": k})
    out_pred = args.prediction_json.replace(".json", "_checked.json")
    with open(out_pred, 'w') as f:
        json.dump(predictions, f, indent=2)
    dataset_name = os.path.basename(args.tiled_h5).replace("_tiled.h5", "")
    report_path = os.path.join(os.path.dirname(args.prediction_json), f"pre_contracted_{dataset_name}.json")
    with open(report_path, 'w') as f:
        json.dump(pre_contracted_flags, f, indent=2)
    print(f"uncertain stimuli checked : {sum(1 for e in predictions if e['prediction'] is None)}")
    print(f"pre-contracted flagged    : {n_flagged}")
    print(f"updated predictions saved → {out_pred}")
    print(f"report saved              → {report_path}")


if __name__ == "__main__":
    main()