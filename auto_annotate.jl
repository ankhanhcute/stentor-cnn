using HDF5, DataFrames

using Pkg

using Images, ImageFiltering, ImageMorphology, ImageContrastAdjustment, ImageSegmentation, Statistics, ProgressBars

using GLMakie

import GeometryOps as GO, GeoInterface as GI

using GeometryBasics
import Meshes


# Load data
begin
    function load_data(fold_name; annotated=true,num_stim=20)
        tiled_path = "$MASTER_FOLDER/data/$fold_name/tiled/$(fold_name)_tiled.h5"
        annot_path = "$MASTER_FOLDER/data/$fold_name/annotated/$(fold_name)_contractions.h5"
        try
            h5open("$tiled_path")
        catch
            run(`datalad get $tiled_path`)
            run(`datalad get $annot_path`)
        end
        data_file = h5open("$MASTER_FOLDER/data/$fold_name/tiled/$(fold_name)_tiled_data.h5")
        row_num = read(data_file["row_num"])
        col_num = read(data_file["col_num"])
        crop_size = read(data_file["crop_size"])
        num_cells = read(data_file["num_cells"])
        vid_file = h5open("$MASTER_FOLDER/data/$fold_name/tiled/$(fold_name)_tiled.h5")
        println(vid_file)
        trial_1 = read(vid_file["tiled_frames_trial_1"])
        # trial_2 = read(vid_file["tiled_frames_trial_2"])
        t1_reshaped = permutedims(reshape(permutedims(trial_1, (3, 2, 1)), (num_stim*2, crop_size, crop_size, num_cells)), (4, 3, 2, 1))
        # t2_reshaped = permutedims(reshape(permutedims(trial_2, (3, 2, 1)), (120, crop_size, crop_size, num_cells)), (4, 3, 2, 1))
        if annotated
            annots = read(h5open("$MASTER_FOLDER/data/$fold_name/annotated/$(fold_name)_contractions.h5")["manual"])
            return Gray{N0f8}.(t1_reshaped ./ 255), annots
        end
        return Gray{N0f8}.(t1_reshaped ./ 255)
    end

    function get_folder(MASTER_FOLDER)
        datasets = sort(readdir("$MASTER_FOLDER/data"))
        recorded_folders = []
        picked_folders = []
        annotated_folders = []
        not_sure = []
        for i in datasets
            if isdir("$MASTER_FOLDER/data/$i/annotated") && length(i) == 19
                push!(annotated_folders, i)
            elseif isdir("$MASTER_FOLDER/data/$i/tiled") && length(i) == 19
                push!(picked_folders, i)
            elseif isfile("$MASTER_FOLDER/data/$i/trial_1/dat.bin") && length(i) == 19
                push!(recorded_folders, i)
            else
                push!(not_sure, i)
            end
        end
        return picked_folders
        # return annotated_folders
    end
    function display_image(tiles, annots; filt=x -> x)
        cell_num = Observable(1)
        stimulus_num = Observable(1)
        f = Figure(size=(1000, 1000))
        Label(f[0, 1:19], @lift("Stimulus $($(stimulus_num)) | Contraction: $(annots[$(stimulus_num),$(cell_num)] == 1 ? "Yes" : "No")"))
        ax1 = Makie.Axis(f[2:10, 1:9])
        image!(ax1, @lift((tiles[$(cell_num), :, :, $(stimulus_num)*2-1])), interpolate=false)
        ax2 = Makie.Axis(f[2:10, 10:19])
        image!(ax2, @lift((tiles[$(cell_num), :, :, $(stimulus_num)*2])), interpolate=false)
        ax3 = Makie.Axis(f[11:20, 1:9])
        image!(ax3, @lift(filt(tiles[$(cell_num), :, :, $(stimulus_num)*2-1])), interpolate=false)
        ax4 = Makie.Axis(f[11:20, 10:19])
        image!(ax4, @lift(filt(tiles[$(cell_num), :, :, $(stimulus_num)*2])), interpolate=false)
        stimulus_slider = Slider(f[1, 1:19], range=1:size(tiles, 4)÷2, startvalue=1)
        cell_slider = Slider(f[2:10, 20], horizontal=false, range=1:size(tiles, 1), startvalue=1)
        on(cell_slider.value) do val
            cell_num[] = val
        end
        on(stimulus_slider.value) do val
            stimulus_num[] = val
        end
        return f, cell_num, stimulus_num
    end
end



# Filter
begin
    function normalize_image(img)
        img = Float16.(img)
        # if maximum(img) > 1.1
        #     img ./= 255
        # end
        img = (img .- mean(img)) ./ (2 * std(img))
        # img = img .+ 1
        # img = clamp.(img, 0, 1)
        return img
    end

    function denoise_image(img; filter_size=9)
        img = imfilter(img, KernelFactors.box(filter_size, filter_size))
        return img
    end

    function enhance_image(img; filter_size=3)
        img = imfilter(img, Kernel.LoG(filter_size))
        return img
    end

    function find_blob(img; filter_size=10)
        img = imfilter(img, Kernel.DoG(filter_size))
        return img
    end

    function threshold_image(img; threshold=0.5)
        img = img .> threshold
        return img
    end

    # Make a simple filter to remove background speckle using closing and opening
    function filter_speckle(img)
        img_closed = closing(img)
        img_opened = opening(img_closed)
        return img_opened
    end

    function final_filter(tiles; threshold=0.6)
        tiles = Float16.(tiles)
        background = mean(tiles, dims=4)
        foreground = background .- tiles
        # norm = stack([stack([normalize_image(foreground[cell,:,:,frame]) for frame in axes(foreground,4)],dims=3) for cell in axes(foreground,1)],dims=1)
        norm_mean = mapslices(mean, foreground, dims=(2, 3))
        norm_std = mapslices(std, foreground, dims=(2, 3))
        norm = (foreground .- norm_mean) ./ (2 * norm_std)
        thresholded = threshold_image(norm, threshold=threshold)
        mask = repeat((thresholded[:, :, :, 1:2:end] .+ thresholded[:, :, :, 2:2:end]) .> 0, inner=(1, 1, 1, 2))
        println(size(mask))
        masked = mask .* foreground
        diff = repeat(norm[:, :, :, 1:2:end] .- norm[:, :, :, 2:2:end], inner=(1, 1, 1, 2))
        return (norm=norm, thresholded=thresholded, mask=mask, diff=diff)
    end
    function process_all_frames(tiles;
        threshold=0.6,
        enhance_filter_size=10,
        enhance_diff_filter_size=3)

        # 1. Final filter (batched over all cells/frames)
        proc = final_filter(tiles; threshold=threshold)

        num_cells = size(tiles, 1)
        num_frames = size(tiles, 4)

        # 2. Threshold masking: normed .* thresholded
        threshed = proc.norm .* proc.thresholded

        # 3. Enhance (LoG, inverted) — broadcast over all cells/frames
        enhanced = similar(threshed)
        for cell in 1:num_cells
            for frame in 1:num_frames
                enhanced[cell, :, :, frame] = -1 .* enhance_image(threshed[cell, :, :, frame]; filter_size=enhance_filter_size)
            end
        end
        # normalize enhanced without mean subtraction using stddev
        enhanced = mapslices(x -> (x ./ (std(x) + 1e-10)), enhanced, dims=(2, 3))

        # 4. Enhanced diff (LoG on masked diff)
        masked_diff = proc.diff .* proc.mask
        enhanced_diff = similar(masked_diff)
        for cell in 1:num_cells
            for frame in 1:num_frames
                enhanced_diff[cell, :, :, frame] = enhance_image(masked_diff[cell, :, :, frame]; filter_size=enhance_diff_filter_size)
            end
        end
        println("Processed $num_cells cells × $num_frames frames")

        return (norm=proc.norm, thresholded=proc.thresholded, mask=proc.mask, diff=proc.diff,
            threshed=threshed, enhanced=enhanced, enhanced_diff=enhanced_diff)
    end
end




function threshold_finder(tiles, annots; filt=x -> x, comp=<)
    rng = 2
    f = Figure(size=(800, 800))
    ax1 = Makie.Axis(f[1, 1])
    ax1.title = "Pixel histogram (normalized)"
    ax2 = Makie.Axis(f[1, 2])
    ax2.title = "Pre-stimulus"
    ax3 = Makie.Axis(f[2, 1])
    ax3.title = "Post-stimulus"
    ax4 = Makie.Axis(f[2, 2])
    ax4.title = "Difference"
    slider = Slider(f[3, :], range=-rng:0.01:rng, startvalue=0.0)
    slider_cell = Slider(f[:, 3], horizontal=false, range=1:size(tiles, 1), startvalue=1)
    slider_stim = Slider(f[4, :], range=1:size(tiles, 4)÷2, startvalue=1)
    Label(f[0, 1:2], @lift("Stimulus $($(slider_stim.value)) | Contraction (manual): $(annots[$(slider_stim.value),$(slider_cell.value)] == 1 ? "Yes" : "No")"))
    filt_img1 = @lift(filt(tiles[$(slider_cell.value), :, :, $(slider_stim.value)*2-1]))
    filt_img2 = @lift(filt(tiles[$(slider_cell.value), :, :, $(slider_stim.value)*2]))
    hist!(ax1, @lift(Float16.($(filt_img1)[:])), bins=-rng:0.1:rng)
    xlims!(ax1, -rng, rng)
    vlines!(ax1, slider.value)
    mask1 = @lift(comp.($(filt_img1), $(slider.value)))
    mask2 = @lift(comp.($(filt_img2), $(slider.value)))
    image!(ax2, @lift(($mask1) .* (filt_img1[])))
    image!(ax3, @lift(($mask2) .* (filt_img2[])))
    # image!(ax2,@lift(tiles[$(slider_cell.value),:,:,$(slider_stim.value)*2-1]))
    # image!(ax3,@lift(tiles[$(slider_cell.value),:,:,$(slider_stim.value)*2]))
    mask = @lift(($mask1 .+ $mask2) .> 0)
    # image!(ax3,@lift($(filt_img) .> $(slider.value)))
    # image!(ax4,mask)
    image!(ax4, @lift(find_blob($(mask) .* (filt_img1[] .- filt_img2[]))), colorrange=[-rng, rng])
    # image!(ax3,filt_img)
    # map(p->p<0.2 ? Gray{N0f8}(0) : p, tiles[1,:,:,1])
    on(events(f).keyboardbutton) do event
        if event.action == Keyboard.press
            if event.key == Keyboard.left
                slider_stim.value[] = max(1, slider_stim.value[] - 1)
            end
            if event.key == Keyboard.right
                slider_stim.value[] = min(size(tiles, 4) ÷ 2, slider_stim.value[] + 1)
            end
        end
    end
    display(GLMakie.Screen(), f)
end


begin #geometry
    function locate_seeds(img; rel_threshold=0.2, max_n=5, take_abs=true)
        if take_abs
            img_abs = abs.(img)
        else
            img_abs = img
        end
        # img_abs ./= maximum(img_abs)
        pts = findlocalmaxima(img_abs)
        pts = pts[img_abs[pts].>rel_threshold]
        inds = partialsortperm(img_abs[pts], 1:min(max_n, length(pts)), rev=true)
        return pts[inds]
    end
    function get_convex_hull(mask)
        inds = findall(mask)                         # CartesianIndex list
        pts = Tuple.(inds)
        # println("Number of points: $(length(pts))")
        if length(pts) < 3
            return Tuple{Int,Int}[]
        end
        hull_poly = GO.convex_hull(pts)
        hull_pts = collect(GI.getpoint(hull_poly))
    end

    function minimum_bounding_rectangle(conv_hull::Vector{Tuple{T,T}}) where {T<:Real}
        n = length(conv_hull)
        if n < 3
            return GeometryBasics.Polygon(Point2f.(conv_hull))
        end
        # calculate edge angles
        #edges = points[ hull_idxs[ 2:end ] ] - points[ hull_idxs[ 1:(end-1) ] ]
        pts = GeometryBasics.Point.(conv_hull)
        edges = pts[1:(end-1)] .- pts[2:end]
        angles = mod.(atan.(first.(edges), last.(edges)), pi / 2)
        angles = unique(angles)
        if isempty(angles)
            return GeometryBasics.Polygon(Point2f.(conv_hull))
        end
        # find rotation matrices
        rot_matrices = zeros(2, 2, length(angles))
        rot_matrices[1, 1, :] = cos.(angles)
        rot_matrices[1, 2, :] = -sin.(angles)
        rot_matrices[2, 1, :] = sin.(angles)
        rot_matrices[2, 2, :] = cos.(angles)
        const_view = hcat(collect.(conv_hull)...)
        rot_points = [rot_matrices[:, :, z] * const_view for z in axes(rot_matrices, 3)]
        # find the bounding points
        min_x = [reduce(min, rp[1, :]) for rp in rot_points]
        max_x = [reduce(max, rp[1, :]) for rp in rot_points]
        min_y = [reduce(min, rp[2, :]) for rp in rot_points]
        max_y = [reduce(max, rp[2, :]) for rp in rot_points]
        # find the box with the best area
        smallest_box = argmin((max_x .- min_x) .* (max_y .- min_y))
        x1, x2 = min_x[smallest_box], max_x[smallest_box]
        y1, y2 = min_y[smallest_box], max_y[smallest_box]
        return GeometryBasics.Polygon(Point2f.(eachcol(rot_matrices[:, :, smallest_box]' * [x1 y1; x1 y2; x2 y2; x2 y1]')))
    end

    function get_all_hulls(reg)
        conv_hulls = [get_convex_hull((labels_map(reg) .== i)) for i in 1:maximum(labels_map(reg))]
        return conv_hulls
    end

    function get_all_rects(conv_hulls)
        rects = [minimum_bounding_rectangle(conv_hull) for conv_hull in conv_hulls]
        return rects
    end
    function filter_regs(conv_hulls, rects; threshold=0.0, max_area=700, min_area=20, box_min=(25, 25), box_max=(125, 125), min_aspect_ratio=0.2, min_width=0, max_length=70)
        box_pts = [box_min, (box_min[1], box_max[2]), box_max, (box_max[1], box_min[2]), box_min]
        box = GI.Polygon([box_pts])
        areas = Float64[]
        filt_inds = findall(eachindex(conv_hulls)) do i
            x = conv_hulls[i]
            isempty(x) && return false
            poly = GI.Polygon([vcat(x, [x[1]])])
            a = GO.area(poly)
            (a > max_area || a < min_area) && return false
            isect = GO.intersection(poly, box; target=GI.PolygonTrait())
            isempty(isect) && return false
            isect_area = sum(GO.area, isect)
            isect_area > GO.area(poly) * threshold || return false
            # # Check aspect ratio and min width using bounding rect
            r = rects[i]
            long_side, short_side = rect_dims(r)
            short_side < min_width && return false
            long_side > max_length && return false
            (short_side / long_side) < min_aspect_ratio && return false
            push!(areas, a)
            return true
        end
        return (areas=areas, rects=rects[filt_inds], hulls=conv_hulls[filt_inds], box=GeometryBasics.Polygon(Point2f.(box_pts)))
    end

    function rect_dims(rect)
        corners = coordinates(rect)
        side1 = norm(corners[2] .- corners[1])
        side2 = norm(corners[3] .- corners[2])
        long_side = max(side1, side2)
        short_side = min(side1, side2)
        return (long_side, short_side)
    end

    function hull_area(hull)
        isempty(hull) && return 0.0
        poly = GI.Polygon([vcat(hull, [hull[1]])])
        return GO.area(poly)
    end

    function print_hull_areas(conv_hulls::Vector{<:Vector})
        for (i, hull) in enumerate(conv_hulls)
            println("Hull $i: $(hull_area(hull)) px²")
        end
    end

    function seeds_to_hulls(seed_img, grow_img, h, w;
        seed_rel_threshold=0.1,
        seed_max_n=5,
        take_abs=false,
        min_area=100,
        max_area=1500,
        box_min=(25, 25),
        box_max=(125, 125),
        min_aspect_ratio=0.2,
        min_width=25)

        empty_filtered = (areas=Float64[], rects=GeometryBasics.Polygon{2,Float32}[],
            hulls=Vector{Tuple}[], box=GeometryBasics.Polygon(Point2f.([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)])))
        empty_result = (seeds=Tuple[], labels=zeros(Int, h, w),
            hulls=Vector{Tuple}[], plot_hulls=GeometryBasics.Polygon{2,Float32}[],
            filtered_plot_hulls=GeometryBasics.Polygon{2,Float32}[],
            rects=GeometryBasics.Polygon{2,Float32}[], filtered=empty_filtered,
            area=0.0, length=0.0, aspect_ratio=0.0)

        seeds = Tuple.(locate_seeds(seed_img;
            rel_threshold=seed_rel_threshold,
            max_n=seed_max_n,
            take_abs=take_abs))

        isempty(seeds) && return empty_result

        # Seeded region growing
        seed_pairs = Tuple.(zip(
            CartesianIndex.(vcat([(1, 1)], seeds)),
            1:(length(seeds)+1)
        ))
        reg = seeded_region_growing(grow_img, seed_pairs)
        labels = labels_map(reg)

        # Convex hulls and rects
        conv_hulls = get_all_hulls(reg)
        all_rects = get_all_rects(conv_hulls)
        hulls = conv_hulls[2:end]
        plot_hulls = GeometryBasics.Polygon.(map(i -> Point2f.(i), hulls))
        rects = all_rects[2:end]

        # Filter
        filtered = filter_regs(conv_hulls, all_rects;
            min_area=min_area, max_area=max_area,
            box_min=box_min, box_max=box_max,
            min_aspect_ratio=min_aspect_ratio, min_width=min_width)
        filtered_plot_hulls = GeometryBasics.Polygon.(map(i -> Point2f.(i), filtered.hulls))

        # Best region metrics
        area = 0.0
        len = 0.0
        asp = 0.0
        if !isempty(filtered.areas)
            best = argmax(filtered.areas)
            area = filtered.areas[best]
            corners = coordinates(filtered.rects[best])
            side1 = norm(corners[2] .- corners[1])
            side2 = norm(corners[3] .- corners[2])
            long_side = max(side1, side2)
            short_side = min(side1, side2)
            len = long_side
            asp = short_side > 0 ? long_side / short_side : Inf
        end

        return (seeds=seeds, labels=labels,
            hulls=hulls, plot_hulls=plot_hulls,
            filtered_plot_hulls=filtered_plot_hulls,
            rects=rects, filtered=filtered,
            area=area, length=len, aspect_ratio=asp)
    end

    function batch_segment_and_filter(processed;
        seed_rel_threshold=0.1,
        seed_max_n=5,
        min_area=20,
        max_area=700,
        box_min=(50, 50),
        box_max=(100, 100),
        min_aspect_ratio=0.2,
        min_width=0,
        max_length=70)

        num_cells = size(processed.enhanced, 1)
        num_frames = size(processed.enhanced, 4)
        h, w = size(processed.enhanced, 2), size(processed.enhanced, 3)

        kw = (seed_rel_threshold=seed_rel_threshold, seed_max_n=seed_max_n,
            min_area=min_area, max_area=max_area,
            box_min=box_min, box_max=box_max,
            min_aspect_ratio=min_aspect_ratio, min_width=min_width)

        # Allocate result matrices for enhanced
        _alloc() = (
            hulls=Matrix{Vector}(undef, num_cells, num_frames),
            plot_hulls=Matrix{Vector{GeometryBasics.Polygon{2,Float32}}}(undef, num_cells, num_frames),
            filt_hulls=Matrix{Vector{GeometryBasics.Polygon{2,Float32}}}(undef, num_cells, num_frames),
            rects=Matrix{Vector}(undef, num_cells, num_frames),
            filtered=Matrix{NamedTuple}(undef, num_cells, num_frames),
            seeds=Matrix{Vector{Tuple}}(undef, num_cells, num_frames),
            labels=Matrix{Matrix}(undef, num_cells, num_frames),
            areas=zeros(num_cells, num_frames),
            lengths=zeros(num_cells, num_frames),
            aspect_ratios=zeros(num_cells, num_frames),
        )
        enh = _alloc()
        ediff = _alloc()

        for cell in ProgressBar(1:num_cells)
            for frame in ProgressBar(1:num_frames)
                # Enhanced pipeline
                r = seeds_to_hulls(
                    processed.enhanced[cell, :, :, frame],
                    processed.norm[cell, :, :, frame],
                    h, w; take_abs=false, kw...)
                enh.seeds[cell, frame] = r.seeds
                enh.labels[cell, frame] = r.labels
                enh.hulls[cell, frame] = r.hulls
                enh.plot_hulls[cell, frame] = r.plot_hulls
                enh.filt_hulls[cell, frame] = r.filtered_plot_hulls
                enh.rects[cell, frame] = r.rects
                enh.filtered[cell, frame] = r.filtered
                enh.areas[cell, frame] = r.area
                enh.lengths[cell, frame] = r.length
                enh.aspect_ratios[cell, frame] = r.aspect_ratio

                # Enhanced diff pipeline
                rd = seeds_to_hulls(
                    processed.enhanced_diff[cell, :, :, frame],
                    processed.enhanced_diff[cell, :, :, frame],
                    h, w; take_abs=true, kw...)
                ediff.seeds[cell, frame] = rd.seeds
                ediff.labels[cell, frame] = rd.labels
                ediff.hulls[cell, frame] = rd.hulls
                ediff.plot_hulls[cell, frame] = rd.plot_hulls
                ediff.filt_hulls[cell, frame] = rd.filtered_plot_hulls
                ediff.rects[cell, frame] = rd.rects
                ediff.filtered[cell, frame] = rd.filtered
                ediff.areas[cell, frame] = rd.area
                ediff.lengths[cell, frame] = rd.length
                ediff.aspect_ratios[cell, frame] = rd.aspect_ratio
            end
            println("Segmented cell $cell / $num_cells")
        end

        return (enhanced=enh, enhanced_diff=ediff)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__

    Pkg.activate("/home/stentor/stentor_light/video_annotation")
    MASTER_FOLDER = "/home/stentor/stentor_light"
    folds = get_folder(MASTER_FOLDER)
    println(DataFrame(dataset=folds))
    println("Pick folder:")
    #folder = folds[parse(Int, readline())]
    folder = folds[1]

    tiles = load_data(folder, annotated=false)

    # f, cell_num, stimulus_num = display_image(tiles, ant, filt=denoise_image)
    # display(f)

    # threshold_finder(mean(tiles,dims=4) .- tiles,annot,filt=normalize_image, comp= >)

    filtered_frames = process_all_frames(tiles)
    # filtered_frames = process_all_frames(tiled_data[:, :, :, 1:20])
    segmented = batch_segment_and_filter(filtered_frames, seed_rel_threshold=0.1)

    begin
        frame_num = Observable(1)
        cell_num = Observable(1)
        f = Figure(size=(800, 800))
        ax = Makie.Axis(f[1, 1], title="BG subtracted")
        ax2 = Makie.Axis(f[1, 2], title="Norm + seeds")
        ax3 = Makie.Axis(f[2, 1], title="Threshed")
        ax4 = Makie.Axis(f[2, 2], title="Enhanced")
        ax5 = Makie.Axis(f[3, 1], title="Masked diff")
        ax6 = Makie.Axis(f[3, 2], title="Enhanced diff + seeds")
        ax7 = Makie.Axis(f[1, 3], title="Enh labels")
        ax8 = Makie.Axis(f[2, 3], title="Enh hulls")
        ax9 = Makie.Axis(f[3, 3], title="Ediff hulls")
        Label(f[0, 1:3], @lift("Cell $($(cell_num)) | Frame $($(frame_num))"))
        slider = Slider(f[4, :], range=1:size(tiles, 4), startvalue=1)
        slider_cell = Slider(f[:, 4], horizontal=false, range=1:size(tiles, 1), startvalue=1)
        on(slider.value) do val
            frame_num[] = val
        end
        on(slider_cell.value) do val
            cell_num[] = val
        end
        on(events(f).keyboardbutton) do event
            if event.action == Keyboard.press
                if event.key == Keyboard.left
                    frame_num[] = max(1, frame_num[] - 1)
                end
                if event.key == Keyboard.right
                    frame_num[] = min(size(tiles, 4), frame_num[] + 1)
                end
                if event.key == Keyboard.down
                    cell_num[] = max(1, cell_num[] - 1)
                end
                if event.key == Keyboard.up
                    cell_num[] = min(size(tiles, 1), cell_num[] + 1)
                end
            end
        end
        bg_sub = mean(tiles, dims=4) .- tiles
        image!(ax, @lift((bg_sub)[$(cell_num), :, :, $(frame_num)]), colorrange=(-4, 4))
        image!(ax2, @lift(filtered_frames.norm[$(cell_num), :, :, $(frame_num)]), colorrange=(-4, 4))
        image!(ax3, @lift(filtered_frames.threshed[$(cell_num), :, :, $(frame_num)]), colorrange=(-4, 4))
        image!(ax4, @lift(filtered_frames.enhanced[$(cell_num), :, :, $(frame_num)]))
        image!(ax5, @lift(filtered_frames.diff[$(cell_num), :, :, $(frame_num)] .* filtered_frames.mask[$(cell_num), :, :, $(frame_num)]), colorrange=(-2, 2))
        image!(ax6, @lift(filtered_frames.enhanced_diff[$(cell_num), :, :, $(frame_num)]))
        # Enhanced seeds on norm
        scatter!(ax2, @lift(Point2f.(segmented.enhanced.seeds[$(cell_num), $(frame_num)])), color=:red, markersize=8)
        # Enhanced diff seeds on enhanced_diff
        scatter!(ax6, @lift(Point2f.(segmented.enhanced_diff.seeds[$(cell_num), $(frame_num)])), color=:red, markersize=8)
        # Enhanced labels + rects
        image!(ax7, @lift(segmented.enhanced.labels[$(cell_num), $(frame_num)]), transparency=true)
        filt = @lift(segmented.enhanced.filtered[$(cell_num), $(frame_num)])
        # poly!(ax7, @lift($(filt).rects), transparency=true, strokecolor=:green, strokewidth=2)
        poly!(ax7, @lift(segmented.enhanced.plot_hulls[$(cell_num), $(frame_num)]), transparency=true, strokecolor=:green, strokewidth=2)
        # Enhanced hulls
        # poly!(ax8, @lift($(filt).box), transparency=true, strokecolor=:green, strokewidth=2)
        # poly!(ax8, @lift(segmented.enhanced.filt_hulls[$(cell_num), $(frame_num)]), transparency=true, strokecolor=:red, strokewidth=2)
        poly!(ax8, @lift(filt[$(cell_num), $(frame_num)]), transparency=true, strokecolor=:red, strokewidth=2)
        poly!(ax8, @lift($(filt).rects), transparency=true, strokecolor=:green, strokewidth=2)
        xlims!(ax8, 0, 150)
        ylims!(ax8, 0, 150)
        # Enhanced diff hulls
        filt_d = @lift(segmented.enhanced_diff.filtered[$(cell_num), $(frame_num)])
        image!(ax9, @lift(segmented.enhanced_diff.labels[$(cell_num), $(frame_num)]))
        # poly!(ax9, @lift($(filt_d).box), transparency=true, strokecolor=:green, strokewidth=2)
        poly!(ax9, @lift(segmented.enhanced_diff.filt_hulls[$(cell_num), $(frame_num)]), transparency=true, strokecolor=:orange, strokewidth=2)
        poly!(ax9, @lift($(filt_d).rects), transparency=true, strokecolor=:green, strokewidth=2)
        xlims!(ax9, 0, 150)
        ylims!(ax9, 0, 150)
        display(f)
    end
end
# denoised = @lift(denoise_image($(threshed);filter_size=9))
# image!(ax4,denoised,colorrange=(-4,4))

# image!(ax4,@lift(find_blob(denoise_image($threshed,filter_size=15))),colorrange=(-2,2))
# image!(ax4,@lift(find_blob($threshed;filter_size=9)),colorrange=(-2,2))





# scatter!(ax7,@lift(Tuple.($top_n_curr)),color=:red)



