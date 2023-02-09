module FullDetectorExample

using KeypointsDetection
using DataFrames
using Plots

@info "Loading Train Dataframe"
trainDataframe = load_train_dataframe();
augmentedDataframe = augment_dataframe(trainDataframe);

@info "Creating Dataset"
X, y = create_train_dataset(augmentedDataframe);
@info "Size of the dataset: $(size(X, 4))"

@info "Starting Training"
net = define_net_lenet_dropout(;dropout_rate=0.2)
train_losses_steps, valid_losses = train_gpu_net!(net, X, y;
    n_epochs=100,
    patience=-1,
    filename="examples/models/traits_model_full.bson"
);
show_losses(train_losses_steps, valid_losses)

end