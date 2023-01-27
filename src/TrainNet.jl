using Flux
using Flux: flatten, params
using Flux.Data: DataLoader
using Flux.Losses: mse

function define_net_simple_feedforward()
    net = Chain(
        flatten,
        Dense(96 * 96, 100),
        relu,
        Dense(30),
    )
    Loss = mse
    net, Loss
end

function train_gpu_net!(net, X, y, Loss)
    gpu_X, gpu_y = X |> gpu, y |> gpu
    gpu_net = net |> gpu
    train_net(gpu_net, gpu_X, gpu_y, Loss; opt=Adam(), batchsize=128, n_epochs=400, file_name="")
end

function train_net!(net, X, y, Loss;
    opt=Adam(), batchsize=128, n_epochs=400, file_name=""
)
    batches = DataLoader((X, y); batchsize, shuffle=true)
    for current_epoch in 1:n_epochs
        Flux.train!(Loss, params(net), batches, opt)
    end
end