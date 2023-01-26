using Images
using DataFrames
using CSV
using Plots

"""
Load and transform the train dataset
"""
function load_train_dataset(path::String=joinpath("data", "training.csv"))
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
        ) ./ 255
    )') => :Image)
    dataframe
end

"""
Show the image and important keypoints.
"""
function show_image(dataset::DataFrame, index::Integer)
    image = dataset[index, :Image]
    plot(image)
end
