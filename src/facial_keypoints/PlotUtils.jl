using .DataFrames
using .Plots

export show_image_augmented, show_image_and_keypoints, show_image_with_gold, show_errors

"""
Create a scatter plot of keypoints in the row `dataframe[index, :]`. Creates lines
between associated keypoints (e.g. eye_inner_corner, eye_center and eye_outer_corner)
"""
function plot_keypoints(dataframe::DataFrame, index::Integer; suffix::String="")
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
                dataframe[index, prefix * dim] for prefix in group
            ])
        end
        if (length(collect(skipmissing(x))) == 0) || (length(collect(skipmissing(y))) == 0)
            continue
        end
        plot!(x, y, m=:o, label=name * suffix)
    end
end

"""
Create a plot with the image in row `dataframe[index, :]` and a scatter
plot of all the keypoints in the row.
"""
function plot_image_and_keypoints(dataframe::DataFrame, index::Integer)
    plt = plot(dataframe[index, :Image])
    plot_keypoints(dataframe, index)
    plt
end

"""
Display the plot of the image in row `dataframe[index, :]` and a scatter
plot of all the keypoints in the row
"""
function show_image_and_keypoints(dataframe::DataFrame, index::Integer)
    plt = plot_image_and_keypoints(dataframe, index)
    gui(plt)
end

"""
Display the plot of the image in row `dataframe[index, :]` and a scatter
plot of all the keypoints in the row, along with all the gold keypoints
from the `goldDataframe[dataframe[index, :TrueIndex], :]`.

E.g. if `dataframe` contains predictions and `goldDataframe` contains true
labels, this function enables viusal comparison of both.

Warning: column `:TrueIndex` is added by method `sort_by_error`, e.g.
to run this method, you need to either manually add column `:TrueIndex`, or
firstly run method `sort_by_error`
"""
function show_image_with_gold(dataframe::DataFrame, index::Integer; goldDataframe::DataFrame=nothing)
    plt = plot_image_and_keypoints(dataframe, index)
    if !isnothing(goldDataframe)
        plot_keypoints(goldDataframe, dataframe[index, :TrueIndex]; suffix="_Gold")
    end
    gui(plt)
end

"""Plot `image` and open `gui` to show it"""
function show_image(image::AbstractArray)
    plt = plot(image)
    gui(plt)
end

"""Plot the same row of different dataframes in a grid, together with associated keypoints"""
function show_image_augmented(dataframes::AbstractArray{DataFrame}, names::AbstractArray{String}, index::Integer)
    plots = []
    for (dataset, name) in zip(dataframes, names)
        plt = plot_image_and_keypoints(dataset, index)
        title!(plt, name)
        push!(plots, plt)
    end
    plt = plot(plots..., legend=false)
    gui(plt)
end

"""
Display the progress of the training,
`train_losses_steps` and `valid_losses` are the outputs of `train_net` or `train_gpu_net`
"""
function show_losses(train_losses_steps, valid_losses, ylims_param=(1e-3, 1e-2))
    tr_len = length(train_losses_steps)
    val_len = length(valid_losses)
    epoch_steps = floor(Int, tr_len / val_len)

    train_losses = train_losses_steps
    valid_losses = repeat(valid_losses, inner=epoch_steps)

    # plot losses
    plt = plot(train_losses, alpha=0.1, color=:dodgerblue3, label="Train Loss")
    plot!(valid_losses, alpha=0.1, color=:rosybrown, label="Validation Loss")

    # plot rolling mean of losses
    plot!(rollmean(train_losses, epoch_steps * 3), color=:blue, label="Train Loss (smoothed)")
    plot!(rollmean(valid_losses, epoch_steps * 3), color=:red, label="Validation Loss (smoothed)")

    xlabel!("Train Steps")
    ylabel!("Loss")
    ylims!(ylims_param)
    gui(plt)
end

"""
Show the Mean Squared Error between predicted labels and gold labels.
`sortedDataframe` should be the output of `sort_by_error`
"""
function show_errors(sortedDataframe::DataFrame)
    plt = plot(sortedDataframe[!, :Error], title="Errors on Individual Samples", xlabel="Sample ID", ylabel="Mean Squared Error")
    gui(plt)
end
