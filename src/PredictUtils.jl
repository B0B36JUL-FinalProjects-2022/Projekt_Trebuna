using GeometryBasics
using NNlib

# https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
function to_grayscale(img)
    img = permutedims(channelview(img), (3, 2, 1))
    reshape(
        2 .* ( (1/3 .* img[:, :, 1]) .+ (img[:, :, 2] .* 1/3) .+ (img[:, :, 3] .* 1/3) ),
        (size(img, 1), size(img, 2), 1, 1)
    )
end

function crop_out_face(img, boundingbox::Vector{Tuple{Int64, Int64}})
    crop_out_face(img, GeometryBasics.Polygon(Point{2, Float32}.(boundingbox)))
end

function crop_out_face(img_orig, boundingbox::GeometryBasics.Polygon)
    points_x = []
    points_y = []
    for line in boundingbox.exterior.points
        push!(points_x, line.points[1][1])
        push!(points_y, line.points[1][2])
    end
    min_x = Int32(minimum(points_x))
    max_x = Int32(maximum(points_x))
    min_y = Int32(minimum(points_y))
    max_y = Int32(maximum(points_y))

    img = img_orig[min_x:max_x, min_y:max_y]
    old_x = min_x
    old_y = min_y
    old_width = max_x - min_x
    old_height = max_y - min_y

    # if any side of the bounding-box is bigger than 96 pixels
    # then we rescale the image so that the biggest side is
    # exactly 96 pixels big, while keeping the aspect ratio
    new_width, new_height = old_width, old_height
    ratio = 1
    if size(img, 1) >= size(img, 2)
        if size(img, 1) > 96
            new_width=96
            ratio = 96 / size(img, 1)
            new_height=floor(Int64, size(img, 2) * ratio)
            img = imresize(img, new_width, new_height)
        end
    elseif size(img, 1) < size(img, 2)
        if size(img, 2) > 96
            ratio = 96 / size(img, 2)
            new_height=96
            new_width=floor(Int64, size(img, 1) * ratio)
            img = imresize(img, new_width, new_height)
        end
    end
    if size(img, 1) < 96 && size(img,2) < 96
        p1 = 96 - size(img, 1)
        p2 = 96 - size(img, 2)
        p11 = Int(floor(p1 / 2))
        p12 = Int(p1 - p11)
        p21 = Int(floor(p2 / 2))
        p22 = Int(p2 - p22)
        img = NNlib.pad_constant(img, (p11, p12, p21, p22), 0, dims=(1, 2))
    else
        if size(img, 1) < 96
            p1 = 96 - size(img, 1)
            p22, p21 = 0, 0
            p11 = Int(floor(p1 / 2))
            p12 = Int(p1 - p11)
            img = NNlib.pad_constant(img, (p11, p12, 0, 0), 0, dims=(1, 2))
        elseif size(img, 2) < 96
            p2 = 96 - size(img, 2)
            p11, p12 = 0, 0
            p21 = Int(floor(p2 / 2))
            p22 = Int(p2 - p21)
            img = NNlib.pad_constant(img, (0, 0, p21, p22), 0, dims=(1, 2))
        end
    end
    convert(Matrix{Float32}, img), old_x, old_y, ratio, p11, p21
end

function preds_to_full(preds, old_x, old_y, ratio, p11, p21)
    new_preds = []
    for p in preds
        x = p[1] - p11
        y = p[2] - p21
        push!(new_preds, (x / ratio + old_x, y / ratio + old_y))
    end
    new_preds
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
    X = withgpu ? X |> gpu : X
    model = withgpu ? net.model |> gpu : net.model
    denormalize(
        model(reshape(X, (96, 96, 1, 1)))
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
