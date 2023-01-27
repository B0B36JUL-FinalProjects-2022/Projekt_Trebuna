using Images
using DataFrames
using CSV
using Plots

"""
Load and transform the train dataset
"""
function load_train_dataframe(path::String=joinpath("data", "training.csv"))
    dataframe = DataFrame(CSV.File(path))
    transform!(dataframe, :Image => ByRow(
        # Image is a string, numbers delimited by space
        # - read by columns, then transpose
        (image) -> Gray.(reshape(
            map(
                (x) -> parse(Int16, x),
                eachsplit(image)
            ),
            (96, 96)
        )' ./ 255
    )) => :Image)
    dataframe
end

function create_train_dataset(dataframe::DataFrame)
    X = transform(
        dataframe,
        :Image => ByRow((image) -> reshape(Float32.(image), 96*96)) => :Image
    )[:, :Image]
    X = reduce(hcat, X)'
    y = Matrix{Float32}(dataframe[:, 1:end - 1])

    # scale to [-1, 1]
    y = ( y .- 48 ) ./ 48

    
    # TODO: some train test split
    # there do exist libraries for that:
    # Lathe.preprocess
    # maybe just permutation of the elements
    # and getting x for training and 1-x for
    # testing is enough
    X, y
end

function plot_keypoints(dataset::DataFrame, index)
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
    prefix_left(group) = [ "left_" * name for name in group]
    prefix_right(group) = [ "right_" * name for name in group]
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
        plot!(x, y, m=:o, label=name)
    end
end

"""
Show the image and important keypoints.
"""
function show_image(dataset::DataFrame, index::Integer)
    image = dataset[index, :Image]
    plt = plot(image)
    plot_keypoints(dataset, index)
    plt
end
