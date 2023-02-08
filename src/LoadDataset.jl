using Images
using .DataFrames
using CSV

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
function load_train_dataframe(path::String=joinpath("data", "training.csv");)
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
end

function create_predict_dataset(dataframe::DataFrame)
    X = transform(
        dataframe,
        :Image => ByRow((image) -> reshape(Float32.(image), 96*96)) => :Image
    )[:, :Image]
    X = reduce(hcat, X)'
    X = reshape(X, (size(X)[1], 96, 96, 1))
    X = permutedims(X, (2, 3, 4, 1))
end

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

function targets_from_dataframe(dataframe::DataFrame, needed_columns)
    needed_columns = filter(needed_columns) do col
        col != "Image"
    end

    y = Matrix{Float32}(dataframe[:, needed_columns])
    y = transpose(y)
    # scale to [-1, 1]
    y = ( y .- 48 ) ./ 48
end

function create_train_dataset(dataframe::DataFrame;
    needed_columns::AbstractArray{String}=names(dataframe)
)
    dataframe = droprows_wo_needed_columns(dataframe, needed_columns)
    X = create_predict_dataset(dataframe)
    y = targets_from_dataframe(dataframe, needed_columns)
    
    X, y
end
