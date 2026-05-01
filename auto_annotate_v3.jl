using Pkg
Pkg.activate(@__DIR__)

using HDF5, VideoIO, GLMakie, Colors, ImageFiltering, ImageMorphology,
      Statistics, DataFrames, ProgressBars, JSON, Images, LinearAlgebra

include(joinpath(@__DIR__, "annotate.jl"))  # annot, plot_fig, lim, clear_plot

const MASTER_FOLDER = abspath(joinpath(@__DIR__, ".."))
const SKIP_FOLDERS  = Set(["2026_04_04_17_25_41"])

# ================================================================
#  I/O  — shared with v2 except tiles are NOT divided by 255
# ================================================================

function list_folders()
    datasets = sort(readdir(joinpath(MASTER_FOLDER, "data")))
    picked, annotated = String[], String[]
    for d in datasets
        length(d) == 19 || continue
        d in SKIP_FOLDERS && continue
        base = joinpath(MASTER_FOLDER, "data", d)
        if isdir(joinpath(base, "annotated"))
            push!(annotated, d)
        elseif isdir(joinpath(base, "tiled"))
            push!(picked, d)
        end
    end
    return picked, annotated
end

function load_tiles_all_trials(fold_name)
    base     = joinpath(MASTER_FOLDER, "data", fold_name)
    tiled_h5 = joinpath(base, "tiled", "$(fold_name)_tiled.h5")
    meta_h5  = joinpath(base, "tiled", "$(fold_name)_tiled_data.h5")

    m  = h5open(meta_h5)
    row_num   = read(m["row_num"])
    col_num   = read(m["col_num"])
    crop_size = read(m["crop_size"])
    num_cells = read(m["num_cells"])
    close(m)

    f  = h5open(tiled_h5)
    trial_keys = sort([k for k in keys(f) if startswith(k, "tiled_frames_trial_")])
    per_trial = []
    for k in trial_keys
        raw = read(f[k])  # (num_cells*crop_size, crop_size, n_frames)
        n_frames = size(raw, 3)
        reshaped = permutedims(reshape(permutedims(raw, (3, 2, 1)),
                    (n_frames, crop_size, crop_size, num_cells)),
                    (4, 3, 2, 1))
        push!(per_trial, reshaped)
    end
    close(f)
    tiles = cat(per_trial...; dims=4)  # (num_cells, H, W, total_frames)

    # v3 fix: tiles are already Float64 in [0,1]. Convert to Float32 without /255.
    tiles_f = Float32.(tiles)

    num_trials = length(trial_keys)
    num_stim = size(tiles_f, 4) ÷ (2 * num_trials)
    return tiles_f, row_num, col_num, crop_size, num_cells, num_trials, num_stim
end

function load_tiled_video(fold_name, num_trials)
    base = joinpath(MASTER_FOLDER, "data", fold_name, "tiled")
    arrs = []
    for t in 1:num_trials
        path = joinpath(base, "$(fold_name)_trial_$(t)_tiled.mp4")
        push!(arrs, Gray.(cat(VideoIO.openvideo(path)...; dims=3)))
    end
    return cat(arrs...; dims=3)
end

function find_gt_path(fold_name; explicit=nothing)
    base = joinpath(MASTER_FOLDER, "data", fold_name)
    if explicit !== nothing
        isfile(explicit) && return explicit
        p = joinpath(base, explicit, "$(fold_name)_contractions.h5")
        isfile(p) && return p
        p2 = joinpath(base, explicit)
        isfile(p2) && return p2
        println("  explicit ground truth '$explicit' not found, falling back to scan")
    end
    isdir(base) || return nothing
    for d in readdir(base)
        endswith(d, "annotated") || continue
        p = joinpath(base, d, "$(fold_name)_contractions.h5")
        isfile(p) && return p
    end
    return nothing
end

function load_ground_truth(fold_name; explicit=nothing)
    p = find_gt_path(fold_name; explicit=explicit)
    p === nothing && return nothing
    println("  ground truth: $p")
    f = h5open(p, "r")
    gt = haskey(f, "manual") ? read(f["manual"]) : nothing
    close(f)
    return gt
end

# ================================================================
#  Module C: Static artifact exclusion
# ================================================================

function disk_mask(h, w, cy, cx, r)
    m = falses(h, w)
    for i in 1:h, j in 1:w
        if (i - cy)^2 + (j - cx)^2 <= r^2
            m[i, j] = true
        end
    end
    return m
end

"""
Detect static artifacts (dish edges, debris) that persist across all frames.

Strategy:
  1. Compute per-pixel minimum brightness across all frames.
     Pixels that are always dark are either holdfast or static artifacts.
  2. Threshold the persistent-darkness map.
  3. Label connected components.
  4. Keep only components that are likely artifacts:
     - touches tile border, OR
     - high aspect ratio (elongated, like a dish edge), OR
     - far from tile center
  5. Protect the holdfast region (small area near center) from exclusion.

Returns: artifact_mask (BitMatrix) — true where artifacts should be excluded.
"""
function detect_static_artifacts(tiles_cell, bg_mean; artifact_k=0.4, holdfast_protect_r=20)
    h, w, nf = size(tiles_cell)
    cy0, cx0 = h / 2, w / 2

    # Darkness map: bg_mean - pixel. High = dark.
    # Minimum across frames: only pixels dark in EVERY frame survive.
    all_darkness = bg_mean .- tiles_cell
    persistent = minimum(all_darkness; dims=3)[:, :, 1]
    persistent_s = imfilter(persistent, Kernel.gaussian(2))

    # Threshold: pixels that are persistently darker than artifact_k * bg_std
    bg_std = std(Float32.(tiles_cell[:, :, 1]))  # rough estimate
    thr = artifact_k * max(bg_std, 0.01f0)
    artifact_raw = persistent_s .> thr

    # Label connected components of persistent dark pixels
    labels = label_components(artifact_raw)
    nlabels = maximum(labels)
    nlabels == 0 && return falses(h, w)

    artifact_mask = falses(h, w)
    for id in 1:nlabels
        inds = findall(labels .== id)
        isempty(inds) && continue
        ys = [Tuple(i)[1] for i in inds]
        xs = [Tuple(i)[2] for i in inds]
        area = length(inds)
        cy_comp = mean(ys)
        cx_comp = mean(xs)
        dist_to_center = sqrt((cy_comp - cy0)^2 + (cx_comp - cx0)^2)

        # Bounding box for aspect ratio
        y_span = maximum(ys) - minimum(ys) + 1
        x_span = maximum(xs) - minimum(xs) + 1
        aspect = max(y_span, x_span) / max(min(y_span, x_span), 1)

        # Does it touch the tile border?
        touches_border = any(ys .== 1) || any(ys .== h) || any(xs .== 1) || any(xs .== w)

        # Classify as artifact if:
        is_artifact = false
        # Long thin line (dish edge): high aspect ratio and touches border
        if aspect > 5 && touches_border
            is_artifact = true
        end
        # Large persistent blob far from center (not the holdfast)
        if area > 200 && dist_to_center > 30 && touches_border
            is_artifact = true
        end
        # Any persistent component touching border with moderate size
        if touches_border && area > 50 && dist_to_center > 25
            is_artifact = true
        end

        if is_artifact
            # Protect the holdfast region: don't exclude pixels near center
            for idx in inds
                y, x = Tuple(idx)
                if (y - cy0)^2 + (x - cx0)^2 > holdfast_protect_r^2
                    artifact_mask[y, x] = true
                end
            end
        end
    end

    # Dilate artifact mask slightly so edges of artifacts don't leak into segmentation
    artifact_mask = dilate(dilate(artifact_mask))
    return artifact_mask
end

# ================================================================
#  Module D: Segmentation
# ================================================================

"""
Find the connected component of `mask` that overlaps `center`, or the nearest
one within `max_dist` pixels. Returns a boolean mask of just that component.
"""
function component_at_center(mask, center; max_dist=20)
    labels = label_components(mask)
    maximum(labels) == 0 && return falses(size(mask))
    cy, cx = center
    iy = clamp(round(Int, cy), 1, size(mask, 1))
    ix = clamp(round(Int, cx), 1, size(mask, 2))
    target = labels[iy, ix]
    if target == 0
        best_id, best_d = 0, Inf
        for id in 1:maximum(labels)
            inds = findall(labels .== id)
            isempty(inds) && continue
            ys = [Tuple(i)[1] for i in inds]
            xs = [Tuple(i)[2] for i in inds]
            d = (mean(ys) - cy)^2 + (mean(xs) - cx)^2
            if d < best_d
                best_d, best_id = d, id
            end
        end
        if best_d > max_dist^2
            return falses(size(mask))
        end
        target = best_id
    end
    return labels .== target
end

"""
Basic threshold mask: foreground where pixel is darker than bg_mean - k*bg_std.
Stentor is dark (low values), background is bright (high values).
Artifact pixels are zeroed out before connected-component analysis.
"""
function threshold_mask(img, bg_mean, bg_std, artifact_mask; k=0.6)
    thr = bg_mean - k * bg_std
    raw = img .< thr
    # Remove artifact pixels before morphological cleanup
    raw .&= .!artifact_mask
    raw = closing(raw)
    raw = opening(raw)
    return raw
end

"""
Score a candidate mask component for plausibility as a stentor body.
Higher score = more likely the real cell.

Redesigned scoring (v3.1):
  - Strongly penalize distance from center
  - Reasonable area range (not too tiny, not too huge)
  - Penalize border contact (artifacts tend to hug borders)
  - Penalize extreme aspect ratios (dish edges are linear)
  - NO bonus for larger area (this caused k=0.3 leak in v3.0)
"""
function score_component(comp_mask, center, crop_size)
    area = sum(comp_mask)
    area < 5 && return -Inf  # too tiny to be real

    h, w = size(comp_mask)
    inds = findall(comp_mask)
    ys = [Tuple(i)[1] for i in inds]
    xs = [Tuple(i)[2] for i in inds]
    cy_comp = mean(ys)
    cx_comp = mean(xs)

    # Distance of component centroid from tile center
    dist_to_center = sqrt((cy_comp - center[1])^2 + (cx_comp - center[2])^2)

    # Area bounds: between 0.1% and 25% of tile
    tile_area = crop_size * crop_size
    area_frac = area / tile_area
    if area_frac > 0.25
        return -Inf  # too large — almost certainly a leak
    end
    if area_frac < 0.0002
        return -Inf  # too tiny (under ~5px for 150px tile)
    end

    # Bounding box aspect ratio
    y_span = maximum(ys) - minimum(ys) + 1
    x_span = maximum(xs) - minimum(xs) + 1
    aspect = max(y_span, x_span) / max(min(y_span, x_span), 1)

    # Border contact: fraction of component pixels on tile edge
    border_pixels = sum(ys .== 1) + sum(ys .== h) + sum(xs .== 1) + sum(xs .== w)
    border_frac = border_pixels / max(area, 1)

    # Compactness: area / bounding-box area. Cell-like blobs are reasonably compact.
    bbox_area = y_span * x_span
    compactness = area / bbox_area

    score = 100.0                           # start with base score
    score -= dist_to_center * 3.0           # strongly prefer close to center
    score -= border_frac * 150.0            # strongly penalize border-huggers
    score -= max(aspect - 4.0, 0.0) * 20.0 # penalize very elongated shapes
    score += compactness * 30.0             # reward compact blobs
    return score
end

# Low-confidence threshold: below this score, flag for human review
const LOW_CONFIDENCE_THRESHOLD = 10.0

"""
Adaptive segmentation: try multiple k values, for each k extract the central
component, and pick the k that produces the best-scoring component.

If `max_area` is provided, components exceeding that area are rejected.
When all k values produce oversized components, the smallest one is returned
and flagged as low confidence.

Returns: (mask, k_used, confidence, low_confidence)
"""
function segment_adaptive(img, bg_mean, bg_std, center, crop_size, artifact_mask;
                          k_values=[0.4, 0.6, 0.8, 1.0, 1.2],
                          max_area=Inf)
    candidates = []

    for k in k_values
        raw_mask = threshold_mask(img, bg_mean, bg_std, artifact_mask; k=k)
        comp = component_at_center(raw_mask, center)
        s = score_component(comp, center, crop_size)
        area = sum(comp)
        push!(candidates, (mask=comp, k=k, score=s, area=Float64(area)))
    end

    # Sort by score descending
    sort!(candidates, by=c -> -c.score)

    # Pick the best candidate that fits under the area cap
    best = nothing
    for c in candidates
        if c.area <= max_area && c.score > -Inf
            best = c
            break
        end
    end

    low_conf = false
    if best === nothing
        # All candidates exceeded max_area — use the smallest one, flag low confidence
        valid = filter(c -> c.area > 0, candidates)
        if isempty(valid)
            return (mask=falses(size(img)), k_used=0.0, confidence=-Inf, low_confidence=true)
        end
        best = valid[argmin([c.area for c in valid])]
        low_conf = true  # area cap forced fallback
    end

    # Flag if no valid component was found at all
    if best.score == -Inf
        low_conf = true
    end

    # Light cleanup on the winning mask
    result_mask = best.mask
    if sum(result_mask) > 0
        result_mask = closing(result_mask)
        result_mask = opening(result_mask)
    end

    return (mask=result_mask, k_used=best.k, confidence=best.score, low_confidence=low_conf)
end

# --- Holdfast (anchor) detection ---

"""
Find skeleton endpoints: pixels in the skeleton that have exactly 1 neighbor
in the 8-connected sense.
"""
function skeleton_endpoints(skel)
    h, w = size(skel)
    endpoints = Tuple{Int,Int}[]
    for y in 1:h, x in 1:w
        skel[y, x] || continue
        neighbors = 0
        for dy in -1:1, dx in -1:1
            (dy == 0 && dx == 0) && continue
            ny, nx = y + dy, x + dx
            if 1 <= ny <= h && 1 <= nx <= w && skel[ny, nx]
                neighbors += 1
            end
        end
        if neighbors <= 1
            push!(endpoints, (y, x))
        end
    end
    return endpoints
end

"""
Segment a single frame for holdfast detection. Returns (mask, score).
"""
function segment_frame_for_holdfast(frame, bg_mean, bg_std, artifact_mask, center, crop_size)
    best_mask = falses(size(frame))
    best_score = -Inf
    for k in [0.4, 0.6, 0.8, 1.0, 1.2]
        thr = bg_mean - k * bg_std
        raw = frame .< thr
        raw .&= .!artifact_mask
        raw = closing(opening(raw))
        comp = component_at_center(raw, center)
        sum(comp) < 20 && continue
        s = score_component(comp, center, crop_size)
        if s > best_score
            best_score = s
            best_mask = comp
        end
    end
    return best_mask, best_score
end

"""
Find the holdfast using the contracted-ball strategy.

Primary strategy (contracted-ball):
  Stentor always contracts TOWARD its holdfast. So the centroid of the
  smallest, most compact post-stimulus mask = the holdfast location.
  1. Segment all post-stimulus frames.
  2. Find the frame with the smallest valid mask near center (= most contracted).
  3. Use that mask's centroid as the holdfast.

Fallback (skeleton):
  If no post frame shows a clear contraction, use the resting mask skeleton:
  find the narrow-end tip closest to center.

Last resort:
  Closest mask pixel to tile center.

Returns: (holdfast, all_ball_positions, method)
  - holdfast: (y, x) of the chosen holdfast
  - all_ball_positions: Vector of (y, x) from every valid contracted ball
    (used by detect_cell_movement to check if the cell relocated)
  - method: String describing which strategy was used
"""
function find_holdfast(tiles_cell, bg_mean, bg_std, artifact_mask, crop_size)
    h, w, nf = size(tiles_cell)
    cy0, cx0 = Float64(h) / 2, Float64(w) / 2
    center = (cy0, cx0)

    # --- Primary: find the most contracted post-stimulus frame ---
    post_indices = 2:2:nf
    min_area_threshold = 20
    max_area_threshold = 80
    candidates = []

    for fi in post_indices
        frame = Float32.(tiles_cell[:, :, fi])
        mask, score = segment_frame_for_holdfast(frame, bg_mean, bg_std, artifact_mask, center, crop_size)
        area = sum(mask)
        area < min_area_threshold && continue
        area > max_area_threshold && continue
        score < LOW_CONFIDENCE_THRESHOLD && continue

        inds = findall(mask)
        ys = [Tuple(i)[1] for i in inds]
        xs = [Tuple(i)[2] for i in inds]
        y_span = maximum(ys) - minimum(ys) + 1
        x_span = maximum(xs) - minimum(xs) + 1
        compactness = area / (y_span * x_span)
        compactness < 0.3 && continue

        cy_m = mean(ys)
        cx_m = mean(xs)
        dist_to_center = sqrt((cy_m - cy0)^2 + (cx_m - cx0)^2)
        dist_to_center > crop_size * 0.3 && continue

        push!(candidates, (cy=Float32(cy_m), cx=Float32(cx_m),
                           area=area, dist=dist_to_center,
                           compactness=compactness, frame=fi, mask=mask))
    end

    all_ball_positions = [(c.cy, c.cx) for c in candidates]

    if !isempty(candidates)
        best = candidates[1]
        best_score = -Inf
        for c in candidates
            area_norm = c.area / max_area_threshold
            dist_norm = c.dist / (crop_size * 0.3)
            s = -area_norm * 0.5 - dist_norm * 4.0 + c.compactness * 0.5
            if s > best_score
                best_score = s
                best = c
            end
        end
        inds = findall(best.mask)
        _, nearest_idx = findmin([(Tuple(i)[1] - cy0)^2 + (Tuple(i)[2] - cx0)^2 for i in inds])
        bp = Tuple(inds[nearest_idx])
        method = "ball(f=$(best.frame),a=$(round(Int, best.area)))"
        return (holdfast=(Float32(bp[1]), Float32(bp[2])),
                all_ball_positions=all_ball_positions, method=method)
    end

    # --- Fallback: skeleton on resting mask ---
    pre_frames = tiles_cell[:, :, 1:2:end]
    baseline = Float32.(median(pre_frames; dims=3)[:, :, 1])
    rest_mask, rest_score = segment_frame_for_holdfast(baseline, bg_mean, bg_std, artifact_mask, center, crop_size)

    if sum(rest_mask) >= 20
        skel = thinning(rest_mask)
        eps = skeleton_endpoints(skel)

        if !isempty(eps)
            dt = distance_transform(feature_transform(.!rest_mask))
            max_dt = maximum(dt[rest_mask])
            best_ep = eps[1]
            best_ep_score = -Inf
            for (ey, ex) in eps
                width_norm = max_dt > 0 ? dt[ey, ex] / max_dt : 0.0
                dist_norm = sqrt((ey - cy0)^2 + (ex - cx0)^2) / (crop_size / 2)
                ep_score = -width_norm * 1.0 - dist_norm * 3.0
                if ep_score > best_ep_score
                    best_ep_score = ep_score
                    best_ep = (ey, ex)
                end
            end
            return (holdfast=(Float32(best_ep[1]), Float32(best_ep[2])),
                    all_ball_positions=all_ball_positions, method="skeleton($(length(eps))ep)")
        end

        inds = findall(rest_mask)
        _, best_idx = findmin([(Tuple(i)[1] - cy0)^2 + (Tuple(i)[2] - cx0)^2 for i in inds])
        bp = Tuple(inds[best_idx])
        return (holdfast=(Float32(bp[1]), Float32(bp[2])),
                all_ball_positions=all_ball_positions, method="nearest_mask_pixel")
    end

    # --- Last resort: tile center ---
    return (holdfast=(Float32(cy0), Float32(cx0)),
            all_ball_positions=all_ball_positions, method="fallback_center")
end

"""
Detect whether a cell moved during the experiment by comparing contracted-ball
positions from early, middle, and late post-stimulus frames.

Small drift is normal. Only flag cells that are totally out of place or swam away.

Strategy:
  1. Divide post frames into early (first third), middle, and late (last third).
  2. Find the median ball position in each window.
  3. If any two windows differ by >move_threshold → cell relocated.
  4. If balls exist in early but not late → cell may have swum away.

Returns: (moved::Bool, spread::Float64, reason::String)
"""
function detect_cell_movement(all_ball_positions, post_frame_indices;
                              move_threshold=40.0)
    if length(all_ball_positions) < 2
        return (moved=false, spread=0.0, reason="")
    end

    n_post = length(post_frame_indices)
    early_cutoff = post_frame_indices[max(1, n_post ÷ 3)]
    late_start   = post_frame_indices[min(n_post, 2 * n_post ÷ 3 + 1)]

    # Match each ball position to its frame index via the order in all_ball_positions
    # (balls are collected in frame order from find_holdfast)
    # We need the frame indices that produced each ball — they were collected
    # from post_frame_indices in order, but not all frames produce valid balls.
    # Since we can't track frame indices here, use position in the list as proxy:
    # first third of balls ≈ early, last third ≈ late.
    n_balls = length(all_ball_positions)
    third = max(1, n_balls ÷ 3)

    early_balls = all_ball_positions[1:third]
    late_balls  = all_ball_positions[max(1, n_balls - third + 1):n_balls]
    mid_balls   = n_balls > 2 ? all_ball_positions[third+1:max(third+1, n_balls-third)] : empty(all_ball_positions)

    function median_pos(balls)
        isempty(balls) && return nothing
        cy = median([b[1] for b in balls])
        cx = median([b[2] for b in balls])
        return (cy, cx)
    end

    early_pos = median_pos(early_balls)
    late_pos  = median_pos(late_balls)
    mid_pos   = median_pos(mid_balls)

    # Check if cell disappeared in late frames
    if early_pos !== nothing && late_pos === nothing
        return (moved=true, spread=Inf, reason="cell absent in late frames")
    end
    if early_pos === nothing && late_pos !== nothing
        return (moved=true, spread=Inf, reason="cell absent in early frames")
    end
    if early_pos === nothing && late_pos === nothing
        return (moved=false, spread=0.0, reason="")
    end

    # Compare positions between windows
    max_shift = 0.0
    reason = ""

    d_el = sqrt((early_pos[1] - late_pos[1])^2 + (early_pos[2] - late_pos[2])^2)
    max_shift = d_el

    if mid_pos !== nothing
        d_em = sqrt((early_pos[1] - mid_pos[1])^2 + (early_pos[2] - mid_pos[2])^2)
        d_ml = sqrt((mid_pos[1] - late_pos[1])^2 + (mid_pos[2] - late_pos[2])^2)
        max_shift = max(d_el, d_em, d_ml)
    end

    moved = max_shift > move_threshold
    if moved
        reason = "relocated $(round(max_shift, digits=1))px between early/late"
    end

    return (moved=moved, spread=max_shift, reason=reason)
end

# ================================================================
#  Feature extraction (per-cell, per-frame)
# ================================================================

function shape_features(mask)
    inds = findall(mask)
    if length(inds) < 5
        return (area=0.0, major_len=0.0, centroid=(NaN32, NaN32))
    end
    coords = reduce(hcat, [collect(Float64.(Tuple(i))) for i in inds])
    μ = mean(coords, dims=2)
    centered = coords .- μ
    C = Symmetric((centered * centered') ./ size(centered, 2))
    evals, _ = eigen(C)
    major_len = 2 * sqrt(max(maximum(evals), 0.0))  # approximate major axis length
    return (area       = Float64(length(inds)),
            major_len  = major_len,
            centroid   = (Float32(μ[1]), Float32(μ[2])))
end

# ================================================================
#  Seeded segmentation — seed-calibrated adaptive threshold
# ================================================================

function build_seeds(img, anchor, artifact_mask, resting_length, crop_size;
                     n_fg_seeds=25, n_bg_per_quadrant=15,
                     guide_mask=nothing)
    h, w = size(img)
    ay, ax = Float64(anchor[1]), Float64(anchor[2])

    radius = max(1.5 * resting_length, crop_size / 3)
    in_disk = falses(h, w)
    for y in 1:h, x in 1:w
        if (y - ay)^2 + (x - ax)^2 <= radius^2
            in_disk[y, x] = true
        end
    end

    fg_region = in_disk .& .!artifact_mask
    fg_pixels = findall(fg_region)
    isempty(fg_pixels) && return nothing, nothing

    fg_values = [img[p] for p in fg_pixels]
    tile_median = median(img)

    order = sortperm(fg_values)
    n_pick = min(n_fg_seeds, length(order))
    fg_inds = fg_pixels[order[1:n_pick]]

    core_mean = mean(fg_values[order[1:n_pick]])
    expand_thr = (core_mean + tile_median) / 2
    fg_set = Set(fg_inds)
    for (i, p) in enumerate(fg_pixels)
        if fg_values[i] < expand_thr && p ∉ fg_set
            push!(fg_inds, p)
            push!(fg_set, p)
        end
    end

    if guide_mask !== nothing && sum(guide_mask) > 0
        guide_pixels = findall(guide_mask .& .!artifact_mask)
        for p in guide_pixels
            if img[p] < tile_median && p ∉ fg_set
                push!(fg_inds, p)
                push!(fg_set, p)
            end
        end
    end

    iay = clamp(round(Int, ay), 1, h)
    iax = clamp(round(Int, ax), 1, w)
    anchor_ci = CartesianIndex(iay, iax)
    if anchor_ci ∉ fg_inds
        push!(fg_inds, anchor_ci)
    end

    bg_region = .!in_disk .& .!artifact_mask
    bg_pixels = findall(bg_region)
    isempty(bg_pixels) && return fg_inds, nothing

    bg_pixels = filter(p -> img[p] >= tile_median, bg_pixels)
    isempty(bg_pixels) && return fg_inds, nothing

    cy, cx = h / 2, w / 2
    quads = [CartesianIndex{2}[] for _ in 1:4]
    for p in bg_pixels
        y, x = Tuple(p)
        q = (y > cy ? 2 : 0) + (x > cx ? 1 : 0) + 1
        push!(quads[q], p)
    end

    bg_inds = CartesianIndex{2}[]
    for q in quads
        isempty(q) && continue
        vals = [img[p] for p in q]
        ord = sortperm(vals; rev=true)
        take = min(n_bg_per_quadrant, length(ord))
        append!(bg_inds, q[ord[1:take]])
    end

    isempty(bg_inds) && return fg_inds, nothing
    return fg_inds, bg_inds
end

function seeded_segment(img, anchor, artifact_mask, resting_length, crop_size;
                        guide_mask=nothing, spatial_prior=nothing,
                        prior_buffer=10, threshold_factor=0.75f0)
    h, w = size(img)
    empty_seeds = CartesianIndex{2}[]

    if spatial_prior !== nothing && sum(spatial_prior) > 0
        search_region = copy(spatial_prior)
        for _ in 1:prior_buffer
            search_region = dilate(search_region)
        end
        search_region .&= .!artifact_mask

        search_pixels = findall(search_region)
        if isempty(search_pixels)
            return (mask=falses(h, w), low_conf=true, reason="empty_search_region")
        end

        intensities = [img[p] for p in search_pixels]
        sorted_int = sort(intensities)
        n_core = min(25, length(sorted_int))
        core_mean = mean(sorted_int[1:n_core])
        tile_median = median(img)
        intensity_thr = core_mean + threshold_factor * (tile_median - core_mean)

        passable = (img .< intensity_thr) .& search_region
        labels = label_components(passable)
        nlabels = maximum(labels)

        if nlabels == 0
            return (mask=falses(h, w), low_conf=true, reason="no_passable_in_prior")
        end

        n_fg = min(50, length(sorted_int))
        fg_inds = search_pixels[sortperm(intensities)[1:n_fg]]

        seed_counts = zeros(Int, nlabels)
        for p in fg_inds
            lbl = labels[p]
            lbl > 0 && (seed_counts[lbl] += 1)
        end
        best_id = argmax(seed_counts)
        mask = seed_counts[best_id] > 0 ? (labels .== best_id) : falses(h, w)
    else
        fg_inds, bg_inds = build_seeds(img, anchor, artifact_mask, resting_length, crop_size;
                                       guide_mask=guide_mask)

        if fg_inds === nothing || isempty(fg_inds)
            return (mask=falses(h, w), low_conf=true, reason="seed_failure")
        end

        fg_intensities = [img[p] for p in fg_inds]
        n_core = min(25, length(fg_intensities))
        core_mean = mean(sort(fg_intensities)[1:n_core])
        tile_median = median(img)
        intensity_thr = core_mean + threshold_factor * (tile_median - core_mean)

        passable = (img .< intensity_thr) .& .!artifact_mask
        labels = label_components(passable)
        nlabels = maximum(labels)

        if nlabels == 0
            return (mask=falses(h, w), low_conf=true, reason="no_passable_region")
        end

        seed_counts = zeros(Int, nlabels)
        for p in fg_inds
            lbl = labels[p]
            lbl > 0 && (seed_counts[lbl] += 1)
        end
        best_id = argmax(seed_counts)
        mask = seed_counts[best_id] > 0 ? (labels .== best_id) : falses(h, w)
    end

    # Shared cleanup
    mask = closing(mask)

    inv_labels = label_components(.!mask)
    border_ids = Set{Int}()
    for i in 1:h
        inv_labels[i, 1] > 0 && push!(border_ids, inv_labels[i, 1])
        inv_labels[i, w] > 0 && push!(border_ids, inv_labels[i, w])
    end
    for j in 1:w
        inv_labels[1, j] > 0 && push!(border_ids, inv_labels[1, j])
        inv_labels[h, j] > 0 && push!(border_ids, inv_labels[h, j])
    end
    for y in 1:h, x in 1:w
        if !mask[y, x] && inv_labels[y, x] > 0 && inv_labels[y, x] ∉ border_ids
            mask[y, x] = true
        end
    end

    area = sum(mask)
    area_too_big = area > 0.5 * h * w
    area_too_small = area < 5

    low_conf = area_too_big || area_too_small
    reason = if area_too_big
        "mask_too_big"
    elseif area_too_small
        "mask_too_small"
    else
        ""
    end

    return (mask=mask, low_conf=low_conf, reason=reason)
end

"""
Segment a post frame using 6 candidates (3 priors × 2 thresholds).
Filter by plausible area, pick the one whose mean intensity is closest
to the pre mask's mean intensity (same cell should have similar darkness).
"""
function segment_post_adaptive(post_img, pre_img, anchor, art_mask, rest_len, crop_size,
                               pre_mask)
    a_pre = sum(pre_mask)
    pre_mean = a_pre > 4 ? mean(pre_img[pre_mask]) : 0.3f0

    anchor_radius = max(rest_len, 25)
    anchor_disk = disk_mask(crop_size, crop_size,
                            Float64(anchor[1]), Float64(anchor[2]),
                            anchor_radius)

    priors = [(pre_mask, "pre"), (anchor_disk, "anchor"), (nothing, "free")]
    thresholds = [0.75f0, 0.90f0]

    candidates = NamedTuple[]
    for (sp, sp_name) in priors
        for tf in thresholds
            if sp !== nothing
                res = seeded_segment(post_img, anchor, art_mask, rest_len, crop_size;
                                     spatial_prior=sp, threshold_factor=tf)
            else
                res = seeded_segment(post_img, anchor, art_mask, rest_len, crop_size;
                                     threshold_factor=tf)
            end
            area = sum(res.mask)
            mi = area > 4 ? mean(post_img[res.mask]) : Inf32
            push!(candidates, (result=res, method="$(sp_name)_$(tf)",
                               area=area, mean_int=mi))
        end
    end

    area_max = min(a_pre * 2, 2000)
    plausible = filter(c -> 100 <= c.area <= area_max, candidates)

    if !isempty(plausible)
        best_idx = argmin([abs(c.mean_int - pre_mean) for c in plausible])
        chosen = plausible[best_idx]
    else
        best_idx = argmin([c.mean_int for c in candidates])
        chosen = candidates[best_idx]
    end

    return chosen.result
end

"""
Compute per-cell features across all frames using seeded segmentation.
Includes Module C (artifact exclusion), movement detection, and confidence flagging.
"""
function compute_features(tiles, crop_size)
    num_cells  = size(tiles, 1)
    num_frames = size(tiles, 4)

    # Per-cell, per-frame storage
    areas      = zeros(Float64, num_cells, num_frames)
    lengths    = zeros(Float64, num_cells, num_frames)
    cy_arr     = fill(NaN32, num_cells, num_frames)
    cx_arr     = fill(NaN32, num_cells, num_frames)
    masks_pre  = Array{BitMatrix}(undef, num_cells, num_frames ÷ 2)
    masks_post = Array{BitMatrix}(undef, num_cells, num_frames ÷ 2)

    resting_area   = zeros(Float64, num_cells)
    resting_length = zeros(Float64, num_cells)
    holdfast       = [(Float32(crop_size / 2), Float32(crop_size / 2)) for _ in 1:num_cells]
    holdfast_method = fill("", num_cells)
    seg_k          = zeros(Float64, num_cells, num_frames)
    seg_conf       = zeros(Float64, num_cells, num_frames)
    low_conf_flags = falses(num_cells, num_frames)
    artifact_masks = Vector{BitMatrix}(undef, num_cells)

    # Per-cell exclusion and movement detection
    excluded       = falses(num_cells)       # true = cell should be excluded
    exclude_reason = fill("", num_cells)
    cell_spread    = zeros(Float64, num_cells)  # max holdfast spread (pixels)

    for c in ProgressBar(1:num_cells)
        pre_frames = tiles[c, :, :, 1:2:end]
        baseline   = Float32.(median(pre_frames; dims=3)[:, :, 1])
        bg_mean    = mean(baseline)
        bg_std     = std(baseline)

        # Module C: detect static artifacts
        all_frames_cell = tiles[c, :, :, :]
        art_mask = detect_static_artifacts(all_frames_cell, bg_mean)
        artifact_masks[c] = art_mask

        # Holdfast detection
        hf_result = find_holdfast(all_frames_cell, bg_mean, bg_std, art_mask, crop_size)
        holdfast[c] = hf_result.holdfast
        holdfast_method[c] = hf_result.method

        # Movement detection: did the cell relocate during the experiment?
        post_indices = collect(2:2:size(all_frames_cell, 3))
        movement = detect_cell_movement(hf_result.all_ball_positions, post_indices)
        cell_spread[c] = movement.spread
        if movement.moved
            excluded[c] = true
            exclude_reason[c] = movement.reason
        end

        # Resting mask from baseline
        anchor = holdfast[c]
        rest_seg = seeded_segment(baseline, anchor, art_mask, 1.0, crop_size)
        rest_feat = shape_features(rest_seg.mask)
        resting_area[c]   = max(rest_feat.area, 1.0)
        resting_length[c] = max(rest_feat.major_len, 1.0)
        rest_len = resting_length[c]

        # Segment all pre frames first (odd frames)
        num_stim = num_frames ÷ 2
        for s in 1:num_stim
            pre_f = 2s - 1
            frame = Float32.(tiles[c, :, :, pre_f])
            seg = seeded_segment(frame, anchor, art_mask, rest_len, crop_size)
            sf = shape_features(seg.mask)
            areas[c, pre_f]    = sf.area
            lengths[c, pre_f]  = sf.major_len
            cy_arr[c, pre_f]   = sf.centroid[1]
            cx_arr[c, pre_f]   = sf.centroid[2]
            seg_k[c, pre_f]    = 0.0
            seg_conf[c, pre_f] = seg.low_conf ? 0.0 : 1.0
            low_conf_flags[c, pre_f] = seg.low_conf
            masks_pre[c, s] = seg.mask
        end

        # Segment post frames using 6-candidate adaptive selection
        for s in 1:num_stim
            post_f = 2s
            pre_f  = 2s - 1
            frame    = Float32.(tiles[c, :, :, post_f])
            pre_frame = Float32.(tiles[c, :, :, pre_f])
            pre_mask = masks_pre[c, s]
            seg = segment_post_adaptive(frame, pre_frame, anchor, art_mask, rest_len,
                                        crop_size, pre_mask)
            sf = shape_features(seg.mask)
            areas[c, post_f]    = sf.area
            lengths[c, post_f]  = sf.major_len
            cy_arr[c, post_f]   = sf.centroid[1]
            cx_arr[c, post_f]   = sf.centroid[2]
            seg_k[c, post_f]    = 0.0
            seg_conf[c, post_f] = seg.low_conf ? 0.0 : 1.0
            low_conf_flags[c, post_f] = seg.low_conf
            masks_post[c, s] = seg.mask
        end
    end

    # Summary
    n_excluded = sum(excluded)
    if n_excluded > 0
        println("  Excluded cells ($n_excluded): $(findall(excluded))")
        for c in findall(excluded)
            println("    cell $c: $(exclude_reason[c])")
        end
    end

    return (areas=areas, lengths=lengths, cy=cy_arr, cx=cx_arr,
            resting_area=resting_area, resting_length=resting_length,
            holdfast=holdfast, holdfast_method=holdfast_method,
            seg_k=seg_k, seg_conf=seg_conf,
            low_conf=low_conf_flags, artifact_masks=artifact_masks,
            masks_pre=masks_pre, masks_post=masks_post,
            excluded=excluded, exclude_reason=exclude_reason,
            cell_spread=cell_spread)
end

# ================================================================
#  Checkpoint visualization — Module D inspection
# ================================================================

"""
Recompute holdfast debug info for visualization.
Shows which method was used (contracted-ball vs skeleton vs fallback)
and the contracted frame if found.
"""
function holdfast_debug_info(tiles_cell, bg_mean, bg_std, artifact_mask, crop_size)
    h, w, nf = size(tiles_cell)
    cy0, cx0 = Float64(h)/2, Float64(w)/2
    center = (cy0, cx0)

    # Check all post frames for contracted ball (same criteria as find_holdfast)
    max_area_ball = 80
    candidates = []
    for fi in 2:2:nf
        frame = Float32.(tiles_cell[:, :, fi])
        mask, score = segment_frame_for_holdfast(frame, bg_mean, bg_std, artifact_mask, center, crop_size)
        area = sum(mask)
        (area < 20 || area > max_area_ball) && continue
        score < LOW_CONFIDENCE_THRESHOLD && continue
        inds = findall(mask)
        ys = [Tuple(i)[1] for i in inds]; xs = [Tuple(i)[2] for i in inds]
        y_span = maximum(ys) - minimum(ys) + 1; x_span = maximum(xs) - minimum(xs) + 1
        compactness = area / (y_span * x_span)
        compactness < 0.3 && continue
        cy_m = mean(ys); cx_m = mean(xs)
        dist = sqrt((cy_m - cy0)^2 + (cx_m - cx0)^2)
        dist > crop_size * 0.3 && continue
        area_norm = area / max_area_ball
        dist_norm = dist / (crop_size * 0.3)
        s = -area_norm * 0.5 - dist_norm * 4.0 + compactness * 0.5
        push!(candidates, (score=s, frame=fi, mask=mask, area=area))
    end

    contracted_frame_idx = nothing
    contracted_mask = nothing
    best_contracted_area = 0
    if !isempty(candidates)
        best = candidates[argmax([c.score for c in candidates])]
        contracted_frame_idx = best.frame
        contracted_mask = best.mask
        best_contracted_area = best.area
    end

    # Also compute skeleton on resting mask
    pre_frames = tiles_cell[:, :, 1:2:end]
    baseline = Float32.(median(pre_frames; dims=3)[:, :, 1])
    rest_mask, _ = segment_frame_for_holdfast(baseline, bg_mean, bg_std, artifact_mask, center, crop_size)
    skel = sum(rest_mask) >= 20 ? thinning(rest_mask) : falses(h, w)
    eps = skeleton_endpoints(skel)

    method = if contracted_frame_idx !== nothing
        "ball(f=$contracted_frame_idx,a=$(round(Int, best_contracted_area)))"
    elseif !isempty(eps)
        "skeleton($(length(eps))ep)"
    else
        "fallback"
    end

    return (rest_mask=rest_mask, skel=skel, endpoints=eps,
            contracted_frame=contracted_frame_idx,
            contracted_mask=contracted_mask, method=method)
end

"""
Save a diagnostic figure for each cell showing:
  Panel 1: Artifact exclusion
  Panel 2: Skeleton + endpoints + holdfast reasoning
  Panel 3: Pre frame with mask overlay
  Panel 4: Post frame with mask overlay

Excluded cells get a dark overlay and "EXCLUDED" label.
Low-confidence frames get a "!" in their panel title.
"""
function save_segmentation_check(fold_name, tiles, feat, crop_size;
                                 max_cells=nothing, stim_idx=1)
    num_cells = size(tiles, 1)
    n = max_cells === nothing ? num_cells : min(max_cells, num_cells)

    out_dir = joinpath(MASTER_FOLDER, "data", fold_name, "v3_checkpoints")
    isdir(out_dir) || mkpath(out_dir)

    println("Saving segmentation check images to $out_dir ...")

    for c in 1:n
        pre_f  = 2 * stim_idx - 1
        post_f = 2 * stim_idx

        pre_img  = tiles[c, :, :, pre_f]
        post_img = tiles[c, :, :, post_f]
        pre_mask  = feat.masks_pre[c, stim_idx]
        post_mask = feat.masks_post[c, stim_idx]
        art_mask  = feat.artifact_masks[c]
        hf_y, hf_x = feat.holdfast[c]
        pre_low  = feat.low_conf[c, pre_f]
        post_low = feat.low_conf[c, post_f]

        is_excluded = feat.excluded[c]
        flag_str = is_excluded ? " EXCLUDED" : (pre_low ? " LOW CONF" : "")

        fig = Figure(size=(1200, 280))

        # Panel 1: Artifact exclusion mask
        ax0 = GLMakie.Axis(fig[1, 1]; title="Artifacts", aspect=DataAspect())
        baseline_img = Float32.(median(tiles[c, :, :, 1:2:end]; dims=3)[:, :, 1])
        image!(ax0, rotr90(baseline_img); colorrange=(0, 1), colormap=:grays)
        art_inds = findall(art_mask)
        if !isempty(art_inds)
            ay = [Tuple(i)[1] for i in art_inds]
            ax_pts = [Tuple(i)[2] for i in art_inds]
            scatter!(ax0, Point2f.(ax_pts, crop_size .+ 1 .- ay); color=RGBAf(1, 0, 0, 0.3), markersize=2)
        end
        hidedecorations!(ax0)

        # Panel 2: Holdfast reasoning
        all_frames_cell = tiles[c, :, :, :]
        bg_mean = mean(baseline_img)
        bg_std = std(baseline_img)
        dbg = holdfast_debug_info(all_frames_cell, bg_mean, bg_std, art_mask, crop_size)
        hf_method = feat.holdfast_method[c]

        hf_title = "Holdfast: $hf_method"
        if is_excluded
            hf_title *= " | $(feat.exclude_reason[c])"
        end
        ax_skel = GLMakie.Axis(fig[1, 2]; title=hf_title, aspect=DataAspect())

        if dbg !== nothing && dbg.contracted_frame !== nothing
            contracted_img = Float32.(tiles[c, :, :, dbg.contracted_frame])
            image!(ax_skel, rotr90(contracted_img); colorrange=(0, 1), colormap=:grays)
            cboundary = dbg.contracted_mask .⊻ erode(dbg.contracted_mask)
            cb_inds = findall(cboundary)
            if !isempty(cb_inds)
                cby = [Tuple(i)[1] for i in cb_inds]
                cbx = [Tuple(i)[2] for i in cb_inds]
                scatter!(ax_skel, Point2f.(cbx, crop_size .+ 1 .- cby); color=:orange, markersize=2)
            end
        else
            image!(ax_skel, rotr90(baseline_img); colorrange=(0, 1), colormap=:grays)
            if dbg !== nothing
                boundary = dbg.rest_mask .⊻ erode(dbg.rest_mask)
                bp_all = findall(boundary)
                if !isempty(bp_all)
                    by, bx = [Tuple(i)[1] for i in bp_all], [Tuple(i)[2] for i in bp_all]
                    scatter!(ax_skel, Point2f.(bx, crop_size .+ 1 .- by); color=RGBAf(0,1,0,0.3), markersize=1)
                end
                skel_inds = findall(dbg.skel)
                if !isempty(skel_inds)
                    sy, sx = [Tuple(i)[1] for i in skel_inds], [Tuple(i)[2] for i in skel_inds]
                    scatter!(ax_skel, Point2f.(sx, crop_size .+ 1 .- sy); color=:yellow, markersize=2)
                end
                for (ey, ex) in dbg.endpoints
                    scatter!(ax_skel, [Point2f(ex, crop_size + 1 - ey)]; color=:white, markersize=8,
                             marker=:diamond)
                end
            end
        end
        scatter!(ax_skel, [Point2f(hf_x, crop_size + 1 - hf_y)]; color=:red, markersize=10)
        scatter!(ax_skel, [Point2f(crop_size/2, crop_size/2)]; color=:cyan, markersize=6,
                 marker=:cross)
        hidedecorations!(ax_skel)

        # Panel 3: Pre frame with mask boundary
        pre_title = "Pre k=$(round(feat.seg_k[c, pre_f], digits=1))" * (pre_low ? " !" : "")
        ax1 = GLMakie.Axis(fig[1, 3]; title=pre_title, aspect=DataAspect())
        image!(ax1, rotr90(pre_img); colorrange=(0, 1), colormap=:grays)
        boundary_pre = pre_mask .⊻ erode(pre_mask)
        bp = findall(boundary_pre)
        if !isempty(bp)
            by, bx = [Tuple(i)[1] for i in bp], [Tuple(i)[2] for i in bp]
            scatter!(ax1, Point2f.(bx, crop_size .+ 1 .- by); color=:lime, markersize=1)
        end
        scatter!(ax1, [Point2f(hf_x, crop_size + 1 - hf_y)]; color=:red, markersize=6)
        hidedecorations!(ax1)

        # Panel 4: Post frame with mask boundary
        post_title = "Post k=$(round(feat.seg_k[c, post_f], digits=1))" * (post_low ? " !" : "")
        ax2 = GLMakie.Axis(fig[1, 4]; title=post_title, aspect=DataAspect())
        image!(ax2, rotr90(post_img); colorrange=(0, 1), colormap=:grays)
        boundary_post = post_mask .⊻ erode(post_mask)
        bp2 = findall(boundary_post)
        if !isempty(bp2)
            by2, bx2 = [Tuple(i)[1] for i in bp2], [Tuple(i)[2] for i in bp2]
            scatter!(ax2, Point2f.(bx2, crop_size .+ 1 .- by2); color=:lime, markersize=1)
        end
        scatter!(ax2, [Point2f(hf_x, crop_size + 1 - hf_y)]; color=:red, markersize=6)
        hidedecorations!(ax2)

        # Info label
        a_pre = round(feat.areas[c, pre_f], digits=0)
        a_post = round(feat.areas[c, post_f], digits=0)
        rest_a = round(feat.resting_area[c], digits=0)
        n_art = sum(art_mask)
        label_color = is_excluded ? :red : (pre_low || post_low ? :orange : :black)
        Label(fig[0, :], "Cell $c | area: $a_pre → $a_post (rest=$rest_a, cap=$(round(Int, 2*rest_a))) | artifacts: $(n_art)px | hf: $hf_method$flag_str",
              color=label_color)

        save(joinpath(out_dir, "seg_cell_$(lpad(string(c), 2, '0'))_stim_$(stim_idx).png"), fig)
    end

    # Summary
    excluded_cells = [c for c in 1:n if feat.excluded[c]]
    flagged_pre = Int[]
    flagged_post = Int[]
    for c in 1:n
        feat.excluded[c] && continue
        pre_f  = 2 * stim_idx - 1
        post_f = 2 * stim_idx
        if feat.low_conf[c, pre_f]
            push!(flagged_pre, c)
        end
        if feat.low_conf[c, post_f]
            push!(flagged_post, c)
        end
    end
    println("Done. Saved $n cell images to $out_dir")
    if !isempty(excluded_cells)
        println("Excluded cells (proposed): $excluded_cells")
        for c in excluded_cells
            println("  cell $c: $(feat.exclude_reason[c])")
        end
    end
    if !isempty(flagged_pre)
        println("Low-confidence (pre-frame issue): $flagged_pre")
    end
    if !isempty(flagged_post)
        println("Post-frame leaks (minor, noted with !): $flagged_post")
    end
    if isempty(excluded_cells) && isempty(flagged_pre)
        println("No excluded or low-confidence cells for stim $stim_idx.")
    end
end

# ================================================================
#  Module H: Contraction detection
# ================================================================

"""
Compute contraction scores for each cell × stimulus pair.

For each pair:
  - logA = log(pre_area / post_area). Positive = cell got smaller.
  - If either frame is low-confidence (area cap fallback, no valid mask),
    the score is NaN (marked NA for human review).

Returns: (scores, calls, na_mask)
  - scores: (total_stim, num_cells) continuous contraction score
  - calls:  (total_stim, num_cells) binary 1=contraction, 0=no, NaN=NA
  - na_mask: (total_stim, num_cells) true where score is unreliable
"""
function compute_contraction_scores(feat; num_stim, num_trials, threshold=0.3)
    num_cells = size(feat.areas, 1)
    total_stim = num_stim * num_trials

    scores  = fill(NaN, total_stim, num_cells)
    calls   = fill(NaN, total_stim, num_cells)
    na_mask = falses(total_stim, num_cells)

    for c in 1:num_cells
        for k in 1:total_stim
            pre_f  = 2k - 1
            post_f = 2k

            # Mark NA if either frame is unreliable
            if feat.low_conf[c, pre_f] || feat.low_conf[c, post_f] || feat.excluded[c]
                na_mask[k, c] = true
                scores[k, c] = NaN
                calls[k, c] = NaN
                continue
            end

            a_pre  = max(feat.areas[c, pre_f],  5.0)
            a_post = max(feat.areas[c, post_f], 5.0)
            scores[k, c] = log(a_pre / a_post)

            calls[k, c] = scores[k, c] > threshold ? 1.0 : 0.0
        end
    end

    return (scores=scores, calls=calls, na_mask=na_mask)
end

"""
Score contraction predictions against ground truth.
"""
function score_contraction(calls, gt)
    valid = .!isnan.(gt) .& .!isnan.(calls)
    sum(valid) == 0 && return nothing
    a = calls[valid] .== 1
    g = gt[valid]   .== 1
    tp = sum(a .&  g)
    fp = sum(a .& .!g)
    fn = sum(.!a .&  g)
    tn = sum(.!a .& .!g)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    n_na = sum(isnan.(calls) .& .!isnan.(gt))
    return (tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, n_na=n_na)
end

"""
Sweep thresholds on logA to find the best F1 against ground truth.
"""
function tune_contraction_threshold(scores, gt)
    valid = .!isnan.(gt) .& .!isnan.(scores)
    sum(valid) == 0 && return (best_threshold=0.3, best_f1=0.0, stats=nothing)

    sv = scores[valid]
    gv = gt[valid] .== 1

    best = (f1=-1.0, threshold=0.3, stats=nothing)
    for τ in 0.0:0.05:2.0
        pred = sv .> τ
        tp = sum(pred .&  gv)
        fp = sum(pred .& .!gv)
        fn = sum(.!pred .&  gv)
        tn = sum(.!pred .& .!gv)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best.f1
            best = (f1=f1, threshold=τ,
                    stats=(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1))
        end
    end
    return (best_threshold=best.threshold, best_f1=best.f1, stats=best.stats)
end

"""
Run contraction detection on a dataset: compute scores, tune threshold
against ground truth if available, print results.
"""
function run_contraction(fold_name; gt_path=nothing)
    println("=== contraction detection :: $fold_name ===")
    tiles, row_num, col_num, crop_size, num_cells, num_trials, num_stim =
        load_tiles_all_trials(fold_name)
    println("  tiles: cells=$num_cells trials=$num_trials stim/trial=$num_stim")

    feat = compute_features(tiles, crop_size)

    # Default threshold
    result = compute_contraction_scores(feat; num_stim=num_stim, num_trials=num_trials)
    n_na = sum(result.na_mask)
    total = num_cells * num_stim * num_trials
    println("  NA pairs: $n_na / $total ($(round(100*n_na/total, digits=1))%)")

    # Score distribution
    valid_scores = filter(!isnan, result.scores[:])
    if !isempty(valid_scores)
        println("  logA scores: min=$(round(minimum(valid_scores), digits=3)) median=$(round(median(valid_scores), digits=3)) max=$(round(maximum(valid_scores), digits=3))")
    end

    gt = load_ground_truth(fold_name; explicit=gt_path)
    if gt !== nothing && size(gt) == size(result.calls)
        println("\n--- default threshold (0.3) ---")
        s = score_contraction(result.calls, gt)
        if s !== nothing
            println("  TP=$(s.tp) FP=$(s.fp) FN=$(s.fn) TN=$(s.tn) | prec=$(round(s.precision, digits=3)) rec=$(round(s.recall, digits=3)) F1=$(round(s.f1, digits=3)) | NA=$(s.n_na)")
        end

        println("\n--- threshold sweep ---")
        tuned = tune_contraction_threshold(result.scores, gt)
        println("  best threshold=$(round(tuned.best_threshold, digits=2)) F1=$(round(tuned.best_f1, digits=3))")
        if tuned.stats !== nothing
            s = tuned.stats
            println("  TP=$(s.tp) FP=$(s.fp) FN=$(s.fn) TN=$(s.tn) | prec=$(round(s.precision, digits=3)) rec=$(round(s.recall, digits=3))")
        end

        # Recompute calls with tuned threshold
        tuned_result = compute_contraction_scores(feat; num_stim=num_stim, num_trials=num_trials,
                                                   threshold=tuned.best_threshold)

        # Show per-cell breakdown for mismatches
        println("\n--- mismatches (tuned threshold=$(round(tuned.best_threshold, digits=2))) ---")
        for c in 1:num_cells
            for k in 1:(num_stim * num_trials)
                isnan(gt[k, c]) && continue
                isnan(tuned_result.calls[k, c]) && continue
                if tuned_result.calls[k, c] != gt[k, c]
                    tag = tuned_result.calls[k, c] == 1 ? "FP" : "FN"
                    println("  cell $c stim $k: $tag (logA=$(round(result.scores[k, c], digits=3)), gt=$(Int(gt[k, c])))")
                end
            end
        end
    elseif gt !== nothing
        println("  ground truth shape $(size(gt)) != scores $(size(result.calls)); skipping validation")
    else
        println("  no ground truth found")
    end

    return (feat=feat, result=result, tiles=tiles, num_cells=num_cells,
            num_stim=num_stim, num_trials=num_trials,
            row_num=row_num, col_num=col_num, crop_size=crop_size)
end

# ================================================================
#  CLI entry point for checkpoint testing
# ================================================================

function run_checkpoint(fold_name; stim_idx=1)
    println("=== auto_annotate_v3 checkpoint :: $fold_name ===")
    tiles, row_num, col_num, crop_size, num_cells, num_trials, num_stim =
        load_tiles_all_trials(fold_name)
    println("  tiles: cells=$num_cells trials=$num_trials stim/trial=$num_stim crop=$crop_size")
    println("  pixel range: $(round(minimum(tiles), digits=3)) .. $(round(maximum(tiles), digits=3))")

    feat = compute_features(tiles, crop_size)
    save_segmentation_check(fold_name, tiles, feat, crop_size; stim_idx=stim_idx)
end

function run_cli()
    if length(ARGS) >= 1 && ARGS[1] == "contraction"
        fold = length(ARGS) >= 2 ? ARGS[2] : error("usage: contraction <folder> [gt_path]")
        gt = length(ARGS) >= 3 ? ARGS[3] : nothing
        run_contraction(fold; gt_path=gt)
    elseif length(ARGS) >= 1
        stim = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
        run_checkpoint(ARGS[1]; stim_idx=stim)
    else
        picked, annotated = list_folders()
        all_folds = vcat(picked, annotated)
        println("Folders:")
        for (i, f) in enumerate(all_folds)
            tag = f in annotated ? "[annotated]" : "[picked]"
            println("  $i  $f  $tag")
        end
        print("Pick #: ")
        idx = parse(Int, readline())
        run_checkpoint(all_folds[idx])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cli()
end
