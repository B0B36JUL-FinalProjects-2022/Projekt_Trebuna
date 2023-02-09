module NoseDetectorExample
using KeypointsDetection
using DataFrames
using Plots

@info "Loading Train Dataframe"
trainDataframe = load_train_dataframe();
augmentedDataframe = augment_dataframe(trainDataframe);

@info "Creating Dataset"
needed_columns=columns_basic_traits
X, y = create_train_dataset(augmentedDataframe; needed_columns);

@info "Starting Training"
net = define_net_lenet_dropout(;n_outputs=length(needed_columns), dropout_rate=0.2)
train_losses_steps, valid_losses = train_gpu_net!(net, X, y;
    n_epochs=100,
    patience=5,
    filename="examples/models/traits_model.bson"
);
show_losses(train_losses_steps, valid_losses)

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