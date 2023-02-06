module NoseDetectorExample
using KeypointsDetection

@info "Loading Train Dataframe"
trainDataframe = KeypointsDetection.load_train_dataframe();
augmentedDataframe = KeypointsDetection.augmentDataframe(trainDataframe);

@info "Creating Dataset"
needed_columns=["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y", "nose_tip_x", "nose_tip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]
X, y = KeypointsDetection.create_train_dataset(augmentedDataframe; needed_columns);

@info "Starting Training"
net = KeypointsDetection.define_net_lenet_dropout(;n_outputs=length(needed_columns), dropout_rate=0.2)
train_losses_steps, valid_losses = KeypointsDetection.train_gpu_net!(net, X, y;
    n_epochs=100,
    patience=5,
    filename="traits_model.bson"
);
KeypointsDetection.plot_losses(train_losses_steps, valid_losses)
end