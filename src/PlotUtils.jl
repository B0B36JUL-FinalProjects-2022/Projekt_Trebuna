using .DataFrames
using .Plots

function plot_keypoints(dataset::DataFrame, index::Int32; suffix::String="")
    eye_group = [
        "eye_inner_corner",
        "eye_center",
        "eye_outer_corner"
    ]
    eyebrow_group = [
        "eyebrow_inner_end",
        "eyebrow_outer_end"
    ]
    mouth_left_right_group = [
        "mouth_left_corner",
        "mouth_right_corner"
    ]
    mouth_top_down_group = [
        "mouth_center_top_lip",
        "mouth_center_bottom_lip"
    ]
    nose_group = ["nose_tip"]
    prefix(group, what) = [what * name for name in group]
    prefix_left(group) = prefix(group, "left_")
    prefix_right(group) = prefix(group, "right_")
    groups = [ 
        (prefix_left(eye_group), "eye_left"),
        (prefix_right(eye_group), "eye_right"),
        (prefix_left(eyebrow_group), "eyebrow_left"),
        (prefix_right(eyebrow_group), "eyebrow_right"),
        (nose_group, "nose"),
        (mouth_left_right_group, "mouth_left_right"),
        (mouth_top_down_group, "mouth_top_bottom")
    ]
    for (group, name) in groups
        x = []
        y = []
        for (dim, arr) in zip(["_x", "_y"], [x, y])
            append!(arr, [
                dataset[index, prefix * dim] for prefix in group
            ])
        end
        if (length(collect(skipmissing(x))) == 0) || (length(collect(skipmissing(y))) == 0)
            continue
        end
        plot!(x, y, m=:o, label=name * suffix)
    end
end

"""
Show the image and important keypoints.
"""
function show_image(dataset::DataFrame, index::Integer; goldDataset::DataFrame=nothing)
    image = dataset[index, :Image]
    plt = plot(image)
    plot_keypoints(dataset, index)
    if !isnothing(goldDataset)
        plot_keypoints(goldDataset, dataset[index, :TrueIndex]; suffix="_Gold")
    end
    gui(plt)
end

function show_image(image::AbstractArray)
    plt = plot(image)
    gui(plt)
end

function show_image_augmented(datasets::AbstractArray{DataFrame}, names::AbstractArray{String}, index::Integer)
    plots = []
    for (dataset, name) in zip(datasets, names)
        plt = show_image(dataset, index)
        title!(plt, name)
        push!(plots, plt)
    end
    plot(plots..., legend=false)
end

function plot_losses(train_losses_steps, valid_losses, ylims_param=(1e-3, 1e-2))
    tr_len = length(train_losses_steps)
    val_len = length(valid_losses)
    epoch_steps = floor(Int, tr_len / val_len)

    train_losses = train_losses_steps
    valid_losses = repeat(valid_losses, inner=epoch_steps)
    plot(train_losses, alpha=0.1, color=:dodgerblue3, label="Train Loss")
    plot!(valid_losses, alpha=0.1, color=:rosybrown, label="Validation Loss")
    plot!(rollmean(train_losses, epoch_steps * 3), color=:blue, label="Train Loss (smoothed)")
    plot!(rollmean(valid_losses, epoch_steps * 3), color=:red, label="Validation Loss (smoothed)")
    xlabel!("Train Steps")
    ylabel!("Loss")
    ylims!(ylims_param)
end

function plot_errors(sortedDataframe::DataFrame)
    plot(sortedDataframe[!, :Error], title="Errors on Individual Samples", xlabel="Sample ID", ylabel="Error")
end
