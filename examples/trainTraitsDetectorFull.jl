module FullDetectorExample

using Revise
using KeypointsDetection
using DataFrames
using FileIO
using Plots
using Serialization

@info "Loading Train Dataframe"
trainDataframe = load_train_dataframe();
augmentedDataframe = augment_dataframe(trainDataframe);

@info "Creating Dataset"
X, y = create_train_dataset(augmentedDataframe);
@info "Size of the dataset: $(size(X, 4))"

@info "Starting Training"
net = define_net_lenet_deeper_dropout(;dropout_rate=0.2)
train_losses_steps, valid_losses = train_gpu_net!(net, X, y;
    n_epochs=100,
    patience=5,
    filename="examples/models/traits_model_full_2.bson"
);
plt = show_losses(train_losses_steps, valid_losses)
savefig(plt, "train_full.png")

# utility functions for this example
function load_im(path) 
    im = open(path, "r") do io
        deserialize(io)
    end
    convert(Matrix{Float32}, im)
end
function predict_to_plot(im)
    preds = predict(net, im; withgpu=true)
    preds = collect(zip(preds[1:2:end], preds[2:2:end]))
    show_gpu_image_scatter(im, preds);
end

function load_and_predict(path)
    im = load_im(path)
    predict_to_plot(im)
    readline()
end


load_and_predict("examples/images/F1.ser")
load_and_predict("examples/images/F2.ser")
load_and_predict("examples/images/F3.ser")
load_and_predict("examples/images/F4.ser")

end