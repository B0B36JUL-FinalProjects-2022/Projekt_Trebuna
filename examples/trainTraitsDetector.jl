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
end