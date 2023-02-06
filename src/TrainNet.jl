
using BSON
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

function forward_pass(net::NetHolder, x)
    net.model(x)
end

function define_net_simple_feedforward(;n_outputs=30)
    net = Chain(
        flatten,
        Dense(96 * 96, 100, relu),
        Dense(100, n_outputs),
    )
    NetHolder(net)
end

function define_net_lenet(;n_outputs=30)
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Conv((5,5), 6=>16, relu, pad=SamePad()),
        MeanPool((2,2)),
        flatten,
        Dense(9216, 120, relu),
        Dense(120, 84, relu),
        Dense(84, n_outputs)
    )
    NetHolder(net)
end

function define_net_lenet_dropout(;dropout_rate=0.5, n_outputs=30)
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        Conv((5,5), 6=>16, relu, pad=SamePad()),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        flatten,
        Dense(9216, 120, relu),
        Dropout(dropout_rate),
        Dense(120, 84, relu),
        Dense(84, n_outputs)
    )
    NetHolder(net)
end

function save_net(net::NetHolder, filename::String)
    model = net.model |> cpu
    BSON.bson(filename, net=model)
end

function load_net(net::NetHolder, filename::String)
    net_weights = BSON.load(filename, @__MODULE__)[:net]
    @info "BSON loaded"
    model = net.model
    Flux.loadparams!(model, params(net_weights))
    net.model = model
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