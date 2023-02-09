
export predict

"""
Use `net` to predict the labels for the image stored in `X`. The image is
reshaped to the right scale.

If flag `withgpu` is set, then both `X` and `net.model` are transferred to gpu
prior to prediction.

The returned predictions are always transferred to the cpu!
"""
function predict(net::NetHolder, X::AbstractArray{Float32, 2}; withgpu=false)
    X = withgpu ? X |> gpu : X
    model = withgpu ? net.model |> gpu : net.model
    Flux.testmode!(model)
    preds = denormalize(
        model(reshape(X, (96, 96, 1, 1)))
    )
    preds = withgpu ? preds |> cpu : preds
    preds
end

"""
Use `net` to predict all the labels for all the images in `X`. If there are more
images than predefined `batch_size` then X is at first batched.

If flag `withgpu` is set, then both `X` and `net.model` are transferred to gpu
prior to prediction.

The returned predictions are always transferred to the cpu!
"""
function predict(net::NetHolder, X::AbstractArray{Float32, 4};
    withgpu=false, batch_size=128
)
    model = withgpu ? net.model |> gpu : net.model
    Flux.testmode!(model)

    # if there are more examples than `batch_size` at first
    # batch all the examples and predict only on the batches
    # to avoid memory overflow on gpu
    if size(X)[end] > batch_size
        batches = DataLoader((X,); batchsize=batch_size, shuffle=false)
        preds = predict_batches(model, batches, withgpu)
    else
        # if there aren't too many samples, predict directly in this method
        X = withgpu ? X |> gpu : X
        preds = model(X)
        preds = withgpu ? preds |> cpu : preds
    end
    denormalize(preds)
end

"""
Traverse all the `batches`, create a big array containing
all the predictions on individual `batches`.

If flag `withgpu` is set, then both `X` and `net.model` are transferred to gpu
prior to prediction.

The returned predictions are always transferred to the cpu!
"""
function predict_batches(model, batches, withgpu)
    Flux.testmode!(model)
    predictions = zeros(Float32, (size(model(first(batches)[1]))[1], 1))
    for batch in batches
        input = batch[1]

        input = withgpu ? input |> gpu : input
        preds = model(input)
        preds = withgpu ? preds |> cpu : preds

        if size(predictions)[2] == 1
            predictions = preds
        else
            predictions = hcat(predictions, preds)
        end
    end
    predictions
end

denormalize(preds) = (preds .* 48) .+ 48