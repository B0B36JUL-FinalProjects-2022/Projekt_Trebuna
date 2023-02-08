using GeometryBasics
using .DataFrames

# https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
function to_grayscale(img)
    reshape(
        (0.299 .* img[:, :, 1]) .+ (img[:, :, 2] .* 0.587) .+ (img[:, :, 3] .* 0.114),
        (size(img, 1), size(img, 2), 1, 1)
    )
end

function predict_from_bb(net::NetHolder, img, boundingboxes::AbstractArray{GeometryBasics.Polygon})
    @info size(img)
    img = to_grayscale(img)
    # model = Chain([Flux.AdaptiveMeanPool((96, 96))])

    # for bb in boundingboxes[2:end]
    #     points_x = []
    #     points_y = []
    #     for line in bb.exterior.points
    #         push!(points_x, line.points[1][1])
    #         push!(points_y, line.points[1][2])
    #     end
    #     min_x = Int32(minimum(points_x))
    #     max_x = Int32(maximum(points_x))
    #     min_y = Int32(minimum(points_y))
    #     max_y = Int32(maximum(points_y))

    #     img2 = img[min_x:max_x, min_y:max_y, :, :]
    #     @show size(img2)
    #     if size(img2, 1) >= 96 && size(img2, 2) >= 96
    #         img2 = model(img2)
    #     elseif size(img2, 1) < 96
    #         #NNLib pad_constant
    #         @info "Not yet implemented!"
    #     elseif size(img1, 1) < 96
    #         @info "Not yet implemented!"
    #     end

        
    #     @show size(img2)
    # end

end

function predict(net::NetHolder, X::AbstractArray{Float32, 2}; withgpu=false)
    Flux.testmode!(net.model)
    denormalize(
        net.model(reshape(X, (96, 96, 1, 1)))
    )
end

BATCH_SIZE=128
function predict(net::NetHolder, X::AbstractArray{Float32, 4}; withgpu=false)
    model = withgpu ? net.model |> gpu : net.model

    if size(X)[end] > BATCH_SIZE
        batches = DataLoader((X,); batchsize=BATCH_SIZE, shuffle=false)
        preds = predict_batches(model, batches, withgpu)
    else
        if withgpu
            X = X |> gpu
        end
        preds = model(X)
        if withgpu
            preds = preds |> cpu
        end
    end
    denormalize(preds)
end

function predict_batches(model, batches, withgpu)
    Flux.testmode!(model)
    predictions = zeros(Float32, (size(model(first(batches)[1]))[1], 1))
    for batch in batches
        input = batch[1]
        if withgpu
            input = input |> gpu
        end
        preds = model(input)
        if size(predictions)[2] == 1
            predictions = preds
        else
            predictions = hcat(predictions, preds)
        end
    end
    predictions
end

denormalize(preds) = (preds .* 48) .+ 48

function predict_to_dataframe(net::NetHolder, dataframe::DataFrame; needed_columns=names(dataframe), withgpu=false)
    X = create_predict_dataset(dataframe)
    if !("Image" in needed_columns)
        push!(needed_columns, "Image")
    end 
    predict_to_dataframe(net, X, dataframe, needed_columns, withgpu)
end

function predict_to_dataframe(net::NetHolder, X::AbstractArray, dataframe::DataFrame, needed_columns, withgpu)
    preds = predict(net, X; withgpu)
    df = DataFrame()
    i = 1
    for name in names(dataframe)
        if name == "Image"
            df[!, name] = dataframe[!, name]
            break
        end
        if name in needed_columns
            df[!, name] = preds[i, :]
            i += 1
        else
            df[!, name] = [missing for _ in range(1, size(preds)[2])]
        end
    end
    df
end