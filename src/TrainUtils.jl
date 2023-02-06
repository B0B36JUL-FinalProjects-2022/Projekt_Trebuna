using BSON

function train_gpu_net!(net::NetHolder, X, y; keywordArgs...)
    gpu_net = NetHolder(net.model |> gpu)
    res = train_net!(gpu_net, X, y; withgpu=true, keywordArgs...)
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
    batchsize=64,
    n_epochs=400,
    valid_size=0.2,
    filename="",
    patience=5,
    withgpu=false
)
    model = net.model
    train_X, train_y, valid_X, valid_y = train_valid_split(X, y; valid_size)
    train_batches = DataLoader((train_X, train_y); batchsize, shuffle=true)
    validation_batches = DataLoader((valid_X, valid_y); batchsize, shuffle=false)
    opt_state = Flux.setup(opt, model)

    train_losses_steps = Float32[]
    valid_losses = Float32[]
    early_stopping = EarlyStopping(patience)
    for epoch in 1:n_epochs
        # train
        epoch_train_losses = Float32[]
        epoch_time = measure_time() do
            Flux.trainmode!(model)
            for batch in train_batches
                train_one_batch!(model, batch, epoch_train_losses, opt_state, loss_fn, withgpu)
            end
            
            epoch_valid_loss = validate(model, validation_batches, loss_fn, withgpu)
            push!(valid_losses, epoch_valid_loss)
        end
        epoch_train_loss = mean(epoch_train_losses)
        @info "[$epoch] TRAIN-LOSS: $epoch_train_loss VALID-LOSS: $(valid_losses[end]) TIME: $epoch_time"
        if should_stop(valid_losses[end], epoch, early_stopping)
            break
        end
        train_losses_steps = vcat(train_losses_steps, epoch_train_losses)
    end
    net.model = model
    !isempty(filename) && save_net(net, filename)
    train_losses_steps, valid_losses
end

mutable struct EarlyStopping
    best_valid::Float32
    best_epoch::Integer
    patience::Integer
    function EarlyStopping(patience::Integer)
        new(Inf32, 0, patience)
    end
end

function should_stop(valid_loss, current_epoch, early_stopping::EarlyStopping)
    if valid_loss <= early_stopping.best_valid
        early_stopping.best_valid = valid_loss
        early_stopping.best_epoch = current_epoch
        return false
    elseif current_epoch > (early_stopping.best_epoch + early_stopping.patience)
        @warn "EARLY STOPPING: EPOCH: $current_epoch"
        return true
    end
    return false
end

function train_valid_split(X::AbstractArray, y::AbstractArray; valid_size)
    ixes = randperm(size(X)[4])
    last_train_ix = floor(Int, length(ixes) * (1 - valid_size))
    train_ixes = ixes[1:last_train_ix]
    valid_ixes = ixes[last_train_ix + 1:end]

    X[:, :, :, train_ixes], y[:, train_ixes], X[:, :, :, valid_ixes], y[:, valid_ixes]
end

function train_one_batch!(model, batch, train_losses_steps, opt_state, loss_fn, withgpu)
    input, label = batch
    if withgpu
        input, label = input |> gpu, label |> gpu
    end

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

function validate(model, validation_batches, loss_fn, withgpu)
    loss, total = 0, 0
    Flux.testmode!(model)
    for batch in validation_batches
        input, label = batch
        if withgpu
            input, label = input |> gpu, label |>gpu
        end
        loss += loss_fn(model(input), label)
        total += 1
    end
    loss / total
end
