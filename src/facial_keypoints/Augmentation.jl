using ImageTransformations
using CoordinateTransformations
using Rotations
using .DataFrames

export horizontal_flip, rotation, augment_dataframe

"""
Traverses the `dataframe` reverses each member of column
`dataframe[:, :Image]` along the `imageDims`, reverses each
member of columns in `indices` and flips columns in `indices` with columns in `new_indices`,
similary for `notIndices` and `new_notIndices`

Args:
    `dataframe::DataFrame` - the dataframe containing at least column `:Image` and columns specified in args ending with `indices`
    `indices::Vector{Integer}` - indices with to be reversed columns
    `notIndices::Vector{Integer}` - indices with not to be reversed columns
    `new_indices::Vector{Integer}` - new columns where `indices` should be stored
    `new_notIndices::Vector{Integer}` - new columns where `notIndices` should be stored
"""
function flip(
    dataframe::DataFrame,
    indices::AbstractArray{<:Integer},
    notIndices::AbstractArray{<:Integer},
    new_indices::AbstractArray{<:Integer},
    new_notIndices::AbstractArray{<:Integer},
    imageDims::Integer
)
    arr_flip(arr) = 96 .- arr
    
    df = transform(dataframe, :Image => ByRow(
        (image) -> reverse(image, dims=imageDims)
        ) => :Image)
    dataframeNames = names(df)

    for (new_i, i) in zip(new_indices, indices)
        df[!, dataframeNames[new_i]] = arr_flip(dataframe[!, dataframeNames[i]])
    end


    for (new_i, i) in zip(new_notIndices, notIndices)
        df[!, dataframeNames[new_i]] = dataframe[!, dataframeNames[i]]
    end

    df
end

"""
Traverses the `dataframe` reverses each member of column
`dataframe[:, :Image]` along the second axis, e.g. flips the image along the center
horizontally, then reverses each member of columns in `indices` and flips columns
in`indices` with columns in `new_indices`, similary for `notIndices` and `new_notIndices`

Args:
- `dataframe::DataFrame`: the dataframe containing at least column `:Image` and columns specified in args ending with `indices`
- `indices::Vector{Integer}`: indices with to be reversed columns
- `notIndices::Vector{Integer}`: indices with not to be reversed columns
- `new_indices::Vector{Integer}`: new columns where `indices` should be stored
- `new_notIndices::Vector{Integer}`: new columns where `notIndices` should be stored

Example:
```julia
> dataframe = DataFrame(
    Image=[
        [1 2; 3 4],
        [1 2; 3 4],
        [1 2; 3 4]
    ],
    A=[10, 20, 30],
    B=[40, 50, 60],
    C=[60, 70, 80]
)
> horizontal_flip(
    dataframe,
    indices=[2, 4],
    notIndices=[3],
    new_indices=[4, 2],
    new_notIndices=[3]
)

... 3×4 DataFrame
 Row │ Image       A      B      C
     │ Array…      Int64  Int64  Int64
─────┼─────────────────────────────────
   1 │ [2 1; 4 3]     36     40     86
   2 │ [2 1; 4 3]     26     50     76
   3 │ [2 1; 4 3]     16     60     66
```
"""
function horizontal_flip(
    dataframe::DataFrame;
    indices = range(1, 30; step=2),
    notIndices = collect(range(2, 31; step=2)),
    new_indices = [
        3, 1, 9, 11, 5, 7, 17, 19, 13, 15,
        21, 25, 23, 27, 29
    ],
    new_notIndices = [
        4, 2, 10, 12, 6, 8, 18, 20, 14, 16,
        22, 26, 24, 28, 30
    ],)
    flip(dataframe, indices, notIndices, new_indices, new_notIndices, 2)
end

"""Replace any NaN in data with zero"""
function replaceNans(data)
    map(data) do x
        isnan(x) ? zero(x) : x
    end
end

"""Rotate the `image` clockwise by `radians`"""
function rotateImage(img, radians)
    parent(replaceNans(warp(
        img,
        CoordinateTransformations.recenter(
            RotMatrix(radians), 
            ImageTransformations.center(img)
        )
    ))[1:96, 1:96])
end

"""
If `value` is greater than `upper_border` or lower than
`lower_border` return `missing` otherwise return the `value`
"""
replace_by_missing(value; lower_border=5, upper_border=91) = (!ismissing(value) && value <= upper_border && value >= lower_border) ? value : missing

"""
Traverses the `dataframe` and rotates each member of `dataframe[:, :Image]` by `radians` clockwise,
also rotates each other column than `:Image` by `radians` and replaces each value that ends up too
close to any border by `missing`
"""
function rotation(dataframe::DataFrame, radians::Float64)
    allowmissing!(dataframe)
    colNames = names(dataframe)
    df = DataFrame(dataframe)
    
    for c in range(1, nrow(dataframe))
        # each image is rotated a bit differently
        noisy_radians = (Random.rand() - 0.5) * 0.2 + radians
        matrix = RotMatrix(noisy_radians)

        # rotate image
        df[c, :Image] = rotateImage(dataframe[c, :Image], noisy_radians)

        # rotate labels
        for i in range(1, 30; step=2)
            vector_x = dataframe[c, colNames[i]]
            vector_y = dataframe[c, colNames[i + 1]]
            vector = hcat(vector_x, vector_y)'

            # center each vector
            vector = vector .- 48

            rotated_vector = matrix * vector

            # move the rotated vector back to center
            rotated_vector = rotated_vector .+ 48
            
            # remove any value too close to the border
            df[c, colNames[i]] = replace_by_missing(rotated_vector[1])
            df[c, colNames[i + 1]] = replace_by_missing(rotated_vector[2])
        end
    end

    df
end

"""
Creates new dataframe by horizontally flipping images, rotation and negative rotation
by pi/5 radians. Hence the returned dataframe is 4-times larger than the original one.
"""
function augment_dataframe(dataframe::DataFrame)
    hf = horizontal_flip(dataframe)
    r = rotation(dataframe, π / 5)
    nr = rotation(dataframe, - π / 5)
    vcat(dataframe, hf, r, nr)
end