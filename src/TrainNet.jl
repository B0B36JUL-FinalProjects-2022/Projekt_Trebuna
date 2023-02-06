
using CUDA
using Flux
using Flux: flatten, params
using Flux.Data: DataLoader
using Flux.Losses: mse
using Plots
using Random
using RollingFunctions
using Statistics

mutable struct NetHolder
    model
end

function define_net_simple_feedforward()
    net = Chain(
        flatten,
        Dense(96 * 96, 100, relu),
        Dense(100, 30),
    )
    NetHolder(net)
end

function define_net_lenet()
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Conv((5,5), 6=>16, relu, pad=SamePad()),
        MeanPool((2,2)),
        flatten,
        Dense(9216, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 30)
    )
    NetHolder(net)
end

function train_valid_split(X::AbstractArray, y::AbstractArray; valid_size)
    ixes = randperm(size(X)[4])
    last_train_ix = floor(Int, length(ixes) * (1 - valid_size))
    train_ixes = ixes[1:last_train_ix]
    valid_ixes = ixes[last_train_ix + 1:end]

    X[:, :, :, train_ixes], y[:, train_ixes], X[:, :, :, valid_ixes], y[:, valid_ixes]
end

function train_one_batch!(model, batch, train_losses_steps, opt_state, loss_fn)
    input, label = batch

    loss, grads = Flux.withgradient(model) do m
        result = m(input)
        loss_fn(result, label)
    end

    # Save the loss from the forward pass. (Done outside of gradient.)
    push!(train_losses_steps, loss)
    
    # Detect loss of Inf or NaN. Print a warning, and then skip update!
    if !isfinite(loss)
        @warn "loss is $loss on item $i" epoch
        return
    end
    Flux.update!(opt_state, model, grads[1])
end

function predict_to_dataframe(net::NetHolder, dataframe::DataFrame)
    X = create_predict_dataset(dataframe)
    predict_to_dataframe(net, X, dataframe)
end

denormalize(preds) = (preds .* 48) .+ 48

function predict_to_dataframe(net::NetHolder, X::AbstractArray, dataframe::DataFrame)
    preds = predict(net, X)
    df = DataFrame()
    for (i, name) in enumerate(names(dataframe))
        if name == "Image"
            df[!, name] = dataframe[!, name]
            break
        end
        df[!, name] = preds[i, :]
    end
    df
end

function predict(net::NetHolder, X::AbstractArray{Float32, 2})
    denormalize(
        net.model(reshape(X, (96, 96, 1, 1)))
    )
end

BATCH_SIZE=128
function predict(net::NetHolder, X::AbstractArray{Float32, 4})
    if size(X)[end] > BATCH_SIZE
        batches = DataLoader((X,); batchsize=BATCH_SIZE, shuffle=false)
        preds = predict_batches(net.model, batches)
    else
        preds = net.model(X)
    end
    denormalize(preds)
end

function predict_batches(model, batches)
    predictions = zeros(Float32, (30, 1))
    for batch in batches
        input = batch[1]
        preds = model(input)
        if size(predictions)[2] == 1
            predictions = preds
        else
            predictions = hcat(predictions, preds)
        end
    end
    predictions
end

function validate(model, validation_batches, loss_fn)
    loss, total = 0, 0
    for batch in validation_batches
        input, label = batch
        loss += loss_fn(model(input), label)
        total += 1
    end
    loss / total
end

function measure_time(method)
    s = time()
    method()
    e = time()
    e - s
end

function train_gpu_net!(net::NetHolder, X, y; keywordArgs...)
    gpu_X, gpu_y = X |> gpu, y |> gpu
    gpu_net = NetHolder(net.model |> gpu)
    res = train_net!(gpu_net, gpu_X, gpu_y; keywordArgs...)
    net.model = gpu_net.model |> cpu
    res
end

"""
Args:
- net

Returns:
- TrainLosses: array with losses calculated on the train set
- ValLosses: array with losses calculated on the validation set"""
function train_net!(net::NetHolder, X, y;
    loss_fn=mse,
    opt=Adam(),
    batchsize=128,
    n_epochs=400,
    valid_size=0.2,
    filename=""
)
    model = net.model
    train_X, train_y, valid_X, valid_y = train_valid_split(X, y; valid_size)
    train_batches = DataLoader((train_X, train_y); batchsize, shuffle=true)
    validation_batches = DataLoader((valid_X, valid_y); batchsize, shuffle=false)
    opt_state = Flux.setup(opt, model)

    train_losses_steps = Float32[]
    valid_losses = Float32[]
    for epoch in 1:n_epochs
        # train
        epoch_train_losses = Float32[]
        epoch_time = measure_time() do
            for batch in train_batches
                train_one_batch!(model, batch, epoch_train_losses, opt_state, loss_fn)
            end
            
            epoch_valid_loss = validate(model, validation_batches, loss_fn)
            push!(valid_losses, epoch_valid_loss)
        end
        epoch_train_loss = mean(epoch_train_losses)
        @info "[$epoch] TRAIN-LOSS: $epoch_train_loss VALID-LOSS: $(valid_losses[end]) TIME: $epoch_time"
        train_losses_steps = vcat(train_losses_steps, epoch_train_losses)
    end
    train_losses_steps, valid_losses
end

function plot_losses(train_losses_steps, valid_losses, ylims_param=(1e-3, 1e-2))
    tr_len = length(train_losses_steps)
    val_len = length(valid_losses)
    epoch_steps = floor(Int, tr_len / val_len)

    train_losses = train_losses_steps
    valid_losses = repeat(valid_losses, inner=epoch_steps)
    plot(train_losses, alpha=0.1, color=:dodgerblue3, label="Train Loss")
    plot!(valid_losses, alpha=0.1, color=:rosybrown, label="Validation Loss")
    plot!(rollmean(train_losses, epoch_steps * 3), color=:blue, label="Train Loss (smoothed)")
    plot!(rollmean(valid_losses, epoch_steps * 3), color=:red, label="Validation Loss (smoothed)")
    xlabel!("Train Steps")
    ylabel!("Loss")
    ylims!(ylims_param)
end