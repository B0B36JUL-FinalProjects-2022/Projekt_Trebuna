using ImageTransformations
using CoordinateTransformations
using Rotations

function flip(
    dataframe::DataFrame,
    indices,
    notIndices,
    new_indices,
    new_notIndices,
    imageDims::Integer
)
    arr_flip(arr) = -1 .* (arr .- 96)
    
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

function horizontalFlip(dataframe::DataFrame)
    indices = range(1, 30; step=2)
    notIndices = collect(range(2, 31; step=2))
    new_indices = [
        3, 1, 9, 11, 5, 7, 17, 19, 13, 15,
        21, 25, 23, 27, 29
    ]
    new_notIndices = [
        4, 2, 10, 12, 6, 8, 18, 20, 14, 16,
        22, 26, 24, 28, 30
    ]
    flip(dataframe, indices, notIndices, new_indices, new_notIndices, 2)
end

function replaceNans(data)
    map(data) do x
        isnan(x) ? zero(x) : x
    end
end

function rotateImage(img, radians)
    parent(replaceNans(warp(
        img,
        CoordinateTransformations.recenter(
            RotMatrix(radians), 
            ImageTransformations.center(img)
        )
    ))[1:96, 1:96])
end

remove_border_values(r) = (!ismissing(r) && r <= 91 && r >= 5) ? r : missing

function rotation(dataframe::DataFrame, radians::Float64)
    allowmissing!(dataframe)
    colNames = names(dataframe)
    df = DataFrame(dataframe)
    
    for c in range(1, nrow(dataframe))
        noisy_radians = (Random.rand() - 0.5) * 0.2 + radians
        matrix = RotMatrix(noisy_radians)
        df[c, :Image] = rotateImage(dataframe[c, :Image], noisy_radians)
        for i in range(1, 30; step=2)
            vector_x = dataframe[c, colNames[i]]
            vector_y = dataframe[c, colNames[i + 1]]
            vector = hcat(vector_x, vector_y)'

            vector = vector .- 48
            rotated_vector = matrix * vector
            rotated_vector = rotated_vector .+ 48
            
            a = remove_border_values(rotated_vector[2])
            df[c, colNames[i]] = remove_border_values(rotated_vector[1])
            df[c, colNames[i + 1]] = remove_border_values(rotated_vector[2])
        end
    end

    df
end

function augmentDataframe(dataframe::DataFrame)
    hf = horizontalFlip(dataframe)
    r = rotation(dataframe, π / 5)
    nr = rotation(dataframe, - π / 5)
    vcat(dataframe, hf, r, nr)
end