
using BSON
using CUDA
using Flux
using Flux: flatten, params
using Flux.Data: DataLoader
using Flux.Losses: mse
using Random

export columns_basic_traits, define_net_simple_feedforward, define_net_lenet, define_net_lenet_dropout, define_net_lenet_deeper_dropout, load_net, save_net

"""
Simple wrapper around the `Flux.Chain` `model`
"""
mutable struct NetHolder
    model
end

"""
Simple feed-forward net with 1 hidden layer with relu activation and the output
without activation. 
"""
function define_net_simple_feedforward(;n_outputs=30)
    net = Chain(
        Flux.flatten,
        Dense(96 * 96, 100, relu),
        Dense(100, n_outputs),
    )
    NetHolder(net)
end

"""
Le-net, e.g. 2 convolutions with stride 1 and same padding, along with
MeanPool operation, the network ends with 3 Dense layer, the last activation
is linear all the previous ones are relu.
"""
function define_net_lenet(;n_outputs=30)
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Conv((5,5), 6=>16, relu, pad=SamePad()),
        MeanPool((2,2)),
        Flux.flatten,
        Dense(9216, 120, relu),
        Dense(120, 84, relu),
        Dense(84, n_outputs)
    )
    NetHolder(net)
end


"""
The same as `define_net_lenet` however between each pair of layers with
parameters there is a dropout layer.

Le-net, e.g. 2 convolutions with stride 1 and same padding, along with
MeanPool operation, the network ends with 3 Dense layer, the last activation
is linear all the previous ones are relu.

`dropout_rate` defines the rate with which the neurons are zeroed out.
"""
function define_net_lenet_dropout(;dropout_rate=0.5, n_outputs=30)
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        Conv((5,5), 6=>16, relu, pad=SamePad()),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        Flux.flatten,
        Dense(9216, 120, relu),
        Dropout(dropout_rate),
        Dense(120, 84, relu),
        Dense(84, n_outputs)
    )
    NetHolder(net)
end

"""
Deeper feedforward convolution network
"""
function define_net_lenet_deeper_dropout(; dropout_rate=0.2, n_outputs=30)
    net = Chain(
        Conv((5,5), 1=>6, relu, pad=SamePad()),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        Conv((5,5), 6=>16, relu),
        MeanPool((2,2)),
        Dropout(dropout_rate),
        Conv((5,5), 16=>32, relu),
        MeanPool((2, 2)),
        Dropout(dropout_rate),
        Conv((5,5), 32=>64, relu),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(256, 128, relu),
        Dropout(dropout_rate),
        Dense(128, n_outputs)
    )
    NetHolder(net)
end

"""
Save `net` to the location specified by `filename`. The method at first
transfers the `net.model` to the cpu
"""
function save_net(net::NetHolder, filename::String)
    model = net.model |> cpu
    BSON.bson(filename, net=model)
end

"""
Load `net` from the location specified by `filename`. The `net` should
be already initialized to the correct structure by one of the net definitions.
This method only loads trained parameters
"""
function load_net(net::NetHolder, filename::String)
    net_weights = BSON.load(filename, @__MODULE__)[:net]
    @info "BSON loaded"
    model = net.model
    Flux.loadparams!(model, params(net_weights))
    net.model = model
end

"""
Columns which we specify as basic traits, e.g. no eyebrows or more specific face traits.
"""
columns_basic_traits=["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y", "nose_tip_x", "nose_tip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]