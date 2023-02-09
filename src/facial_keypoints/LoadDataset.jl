using Images
using .DataFrames
using CSV

export load_train_dataframe, create_train_dataset, create_predict_dataset, targets_from_dataframe

"""
Convert string representation of the image into a matrix with `shape`
"""
function string_to_matrix(str::String; shape=(96, 96))
    reshape(
        map(
            (x) -> parse(Int16, x),
            eachsplit(str)
        ),
        shape
    )'
end

"""
Load the train dataframe from the csv file in `path`,
transform the `:Image` column from string to a `Matrix` of `Gray`
elements
"""
function load_train_dataframe(path::String=joinpath("data", "training.csv");)
    dataframe = DataFrame(CSV.File(path))
    transform!(dataframe, :Image => ByRow(
        # Image is a string, numbers delimited by space
        # - read by columns, then transpose
        # by default convert to Gray in order to be
        # easily displayable as an image
        (str_image) -> Gray.(
            string_to_matrix(str_image) ./ 255
        )
    ) => :Image)
end

"""
Create an array X with shape `(h, w, c, N)`, where `N` is the number
of examples in the dataset and `(h, w, c)` is the shape of the examples
"""
function create_predict_dataset(dataframe::DataFrame)
    X = transform(
        dataframe,
        :Image => ByRow((image) -> reshape(Float32.(image), 96*96)) => :Image
    )[:, :Image]
    X = reduce(hcat, X)'
    X = reshape(X, (size(X)[1], 96, 96, 1))
    X = permutedims(X, (3, 2, 4, 1))
end

"""
Creates a new dataframe which doesn't contain any row where any column belonging to `needed_columns` is missing
"""
function droprows_wo_needed_columns(dataframe::DataFrame, needed_columns::AbstractArray{String})
    filter(dataframe) do row
        for colname in needed_columns
            if ismissing(row[colname])
                return false
            end
        end
        true
    end
end

"""
Create an array with shape `(length(needed_columns), N)` where `N` is the number of examples
in the dataframe.
"""
function targets_from_dataframe(dataframe::DataFrame, needed_columns::AbstractArray{String})
    needed_columns = filter(needed_columns) do col
        col != "Image"
    end

    y = Matrix{Float32}(dataframe[:, needed_columns])
    y = transpose(y)
    # scale to [-1, 1]
    y = ( y .- 48 ) ./ 48
end

"""
Create train inputs and targets with shape `(input_shape..., N)`, `(target_shape..., N)` where
`N` is the number of examples in the dataframe.
"""
function create_train_dataset(dataframe::DataFrame;
    needed_columns::AbstractArray{String}=names(dataframe)
)
    dataframe = droprows_wo_needed_columns(dataframe, needed_columns)
    X = create_predict_dataset(dataframe)
    y = targets_from_dataframe(dataframe, needed_columns)
    
    X, y
end
