import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, binary_closing, binary_opening
from skimage.measure import label 

def score_component(comp_mask, center, crop_size):
    area = np.sum(comp_mask)
    if area < 5:
        return -np.inf
    ys, xs = np.where(comp_mask)
    cy_comp, cx_comp = np.mean(ys), np.mean(xs)
    dist_to_center = np.sqrt((cy_comp - center[0])**2 + (cx_comp - center[1])**2)
    tile_area = crop_size * crop_size
    area_frac = area / tile_area
    if area_frac > 0.25:
        return -np.inf
    if area_frac < 0.0002:
        return -np.inf
    y_span = ys.max() - ys.min() + 1
    x_span = xs.max() - xs.min() + 1
    aspect = max(y_span, x_span) / max(min(y_span, x_span), 1)
    border_pixels = (np.sum(ys == 0) + np.sum(ys == comp_mask.shape[0]-1) +
                     np.sum(xs == 0) + np.sum(xs == comp_mask.shape[1]-1))
    border_frac = border_pixels / max(area, 1)
    compactness = area / (y_span * x_span)
    score = 100.0
    score -= dist_to_center * 3.0
    score -= border_frac * 150.0
    score -= max(aspect - 4.0, 0.0) * 20.0
    score += compactness * 30.0
    return score
def component_at_center(mask, center, max_dist=20):
    labeled = label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    
    cy, cx = center
    iy = int(np.clip(round(cy), 0, mask.shape[0]-1))
    ix = int(np.clip(round(cx), 0, mask.shape[1]-1))
    
    target = labeled[iy, ix]
    if target == 0:
        best_id, best_d = 0, np.inf
        for cid in range(1, labeled.max()+1):
            ys, xs = np.where(labeled == cid)
            if len(ys) == 0:
                continue
            d = (np.mean(ys) - cy)**2 + (np.mean(xs) - cx)**2
            if d < best_d:
                best_d, best_id = d, cid
        if best_d > max_dist**2:
            return np.zeros_like(mask, dtype=bool)
        target = best_id
    
    return labeled == target
def segment_frame_for_holdfast(frame, bg_mean, bg_std, artifact_mask, center, crop_size):
    best_mask = np.zeros(frame.shape, dtype=bool)
    best_score = -np.inf
    
    for k in [0.4, 0.6, 0.8, 1.0, 1.2]:
        thr = bg_mean - k * bg_std
        raw = frame < thr
        raw = raw & ~artifact_mask
        raw = binary_closing(raw)
        raw = binary_opening(raw)
        comp = component_at_center(raw, center)
        if np.sum(comp) < 20:
            continue
        s = score_component(comp, center, crop_size)
        if s > best_score:
            best_score = s
            best_mask = comp
    
    return best_mask, best_score
def skeleton_endpoints(skel):
    endpoints = []
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx]:
                        neighbors += 1
        if neighbors <= 1:
            endpoints.append((y, x))
    return endpoints
def find_holdfast(tiles_cell, crop_size):
    h, w, nf = tiles_cell.shape
    cy0, cx0 = h / 2.0, w / 2.0
    center = (cy0, cx0)

    bg_mean = np.mean(tiles_cell)
    bg_std = np.std(tiles_cell)
    artifact_mask = np.zeros((h, w), dtype=bool)

    # find holdfast through the center of contracted cell
    post_indices = range(1, nf, 2) # look through post-frame to look for ball-like shape
    candidates = []

    for fi in post_indices:
        frame = tiles_cell[:, :, fi].astype(np.float32)
        mask, score = segment_frame_for_holdfast(frame, bg_mean, bg_std, artifact_mask, center, crop_size)
        area = np.sum(mask)
        if area < 20 or area > 80:
            continue
        if score < 10.0:
            continue
        ys, xs = np.where(mask)
        y_span = ys.max() - ys.min() + 1
        x_span = xs.max() - xs.min() + 1
        compactness = area / (y_span * x_span)
        if compactness < 0.3:
            continue
        cy_m, cx_m = np.mean(ys), np.mean(xs)
        dist = np.sqrt((cy_m - cy0)**2 + (cx_m - cx0)**2)
        if dist > crop_size * 0.3:
            continue
        candidates.append({
            'cy': float(cy_m), 'cx': float(cx_m),
            'area': area, 'dist': dist,
            'compactness': compactness,
            'frame': fi, 'mask': mask
        })

    all_ball_positions = [(c['cy'], c['cx']) for c in candidates]

    if candidates:
        best = candidates[0]
        best_score = -np.inf
        for c in candidates:
            area_norm = c['area'] / 80
            dist_norm = c['dist'] / (crop_size * 0.3)
            s = -area_norm * 0.5 - dist_norm * 4.0 + c['compactness'] * 0.5
            if s > best_score:
                best_score = s
                best = c
        ys, xs = np.where(best['mask'])
        dists = (ys - cy0)**2 + (xs - cx0)**2
        idx = np.argmin(dists)
        return {
            'holdfast': (float(ys[idx]), float(xs[idx])),
            'all_ball_positions': all_ball_positions,
            'method': f"ball(f={best['frame']},a={int(best['area'])})"
        }

    # If a cell never contracts, it does not really matter whether or not we find the holdfast
    # Defaulting to finding the end point of elongated cell
    pre_frames = tiles_cell[:, :, 0::2]
    baseline = np.median(pre_frames, axis=2).astype(np.float32)
    rest_mask, _ = segment_frame_for_holdfast(baseline, bg_mean, bg_std, artifact_mask, center, crop_size)

    if np.sum(rest_mask) >= 20:
        skel = skeletonize(rest_mask)
        eps = skeleton_endpoints(skel)
        if eps:
            dt = ndimage.distance_transform_edt(rest_mask)
            max_dt = dt[rest_mask].max() if rest_mask.any() else 1.0
            best_ep = eps[0]
            best_ep_score = -np.inf
            for (ey, ex) in eps:
                width_norm = dt[ey, ex] / max_dt if max_dt > 0 else 0.0
                dist_norm = np.sqrt((ey - cy0)**2 + (ex - cx0)**2) / (crop_size / 2)
                ep_score = -width_norm * 1.0 - dist_norm * 3.0
                if ep_score > best_ep_score:
                    best_ep_score = ep_score
                    best_ep = (ey, ex)
            return {
                'holdfast': (float(best_ep[0]), float(best_ep[1])),
                'all_ball_positions': all_ball_positions,
                'method': f"cell_length"
            }
        ys, xs = np.where(rest_mask)
        dists = (ys - cy0)**2 + (xs - cx0)**2
        idx = np.argmin(dists)
        return {
            'holdfast': (float(ys[idx]), float(xs[idx])),
            'all_ball_positions': all_ball_positions,
            'method': 'nearest_mask_pixel'
        }

    # return the center coordinates if no threshold works, or images are too blurry (can add the confident flag here)
    return {
        'holdfast': (cy0, cx0),
        'all_ball_positions': all_ball_positions,
        'method': 'fallback_center'
    }
if __name__ == "__main__":
    import sys, h5py, matplotlib.pyplot as plt
    tiled_path = sys.argv[1]
    meta_path = sys.argv[2]

    with h5py.File(meta_path, 'r') as m:
        crop_size = int(m['crop_size'][()])
        num_cells = int(m['num_cells'][()])

    with h5py.File(tiled_path, 'r') as f:
        trial_keys = sorted([k for k in f.keys() if k.startswith('tiled_frames_trial_')])
        raw = np.concatenate([f[k][:] for k in trial_keys], axis=0)

    print(f"Loaded: {num_cells} cells, {raw.shape[0]} frames")

    fig, axes = plt.subplots(6, 2, figsize=(8, 18))
    for cell_idx in range(6):
        tiles_cell = raw[:, :, cell_idx*crop_size:(cell_idx+1)*crop_size]
        tiles_cell = np.transpose(tiles_cell, (1, 2, 0)).astype(np.float32)
        result = find_holdfast(tiles_cell, crop_size)
        hy, hx = result['holdfast']
        print(f"Cell {cell_idx} | holdfast=({hy:.1f},{hx:.1f}) | {result['method']}")
        axes[cell_idx,0].imshow(tiles_cell[:,:,0], cmap='gray')
        axes[cell_idx,0].plot(hx, hy, 'r+', markersize=15, markeredgewidth=2)
        axes[cell_idx,0].set_title(f'cell {cell_idx} pre')
        axes[cell_idx,1].imshow(tiles_cell[:,:,1], cmap='gray')
        axes[cell_idx,1].plot(hx, hy, 'r+', markersize=15, markeredgewidth=2)
        axes[cell_idx,1].set_title(f'cell {cell_idx} post | {result["method"]}')
    plt.tight_layout()
    plt.savefig('outputs/holdfast_test.png')
    print("saved → outputs/holdfast_test2.png")