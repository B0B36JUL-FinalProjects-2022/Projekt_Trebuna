using Images
using DataFrames
using CSV
using Plots

function string_to_matrix(str)
    reshape(
        map(
            (x) -> parse(Int16, x),
            eachsplit(str)
        ),
        (96, 96)
    )
end

"""
Load and transform the train dataset
"""
function load_train_dataframe(path::String=joinpath("data", "training.csv"))
    dataframe = DataFrame(CSV.File(path))
    transform!(dataframe, :Image => ByRow(
        # Image is a string, numbers delimited by space
        # - read by columns, then transpose
        # by default convert to Gray in order to be
        # easily displayable as an image
        (str_image) -> Gray.(
            string_to_matrix(str_image)' ./ 255
        )
    ) => :Image)
    dropmissing(dataframe)
end

function create_predict_dataset(dataframe::DataFrame)
    X = transform(
        dataframe,
        :Image => ByRow((image) -> reshape(Float32.(image), 96*96)) => :Image
    )[:, :Image]
    X = reduce(hcat, X)'
    X = reshape(X, (size(X)[1], 96, 96))
    X = permutedims(X, (2, 3, 1))
end


function create_train_dataset_full(dataframe::DataFrame)
    dropmissing!(dataframe)
    X = create_predict_dataset(dataframe)

    y = Matrix{Float32}(dataframe[:, 1:end - 1])
    y = transpose(y)

    # scale to [-1, 1]
    y = ( y .- 48 ) ./ 48
    
    X, y
end
