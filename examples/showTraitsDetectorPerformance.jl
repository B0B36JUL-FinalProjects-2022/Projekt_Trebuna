module TraitsDetectorPerformance
using KeypointsDetection

@info "Loading Dataset"
trainDataframe = KeypointsDetection.load_train_dataframe();
augmentedDataframe = KeypointsDetection.augmentDataframe(trainDataframe);

model_path="traits_model.bson"
@info "Loading Trained Model $(model_path)"
needed_columns=["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y", "nose_tip_x", "nose_tip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]
net = KeypointsDetection.define_net_lenet_dropout(;n_outputs=length(needed_columns), dropout_rate=0.2)
KeypointsDetection.load_net(net, model_path)

# show some prediction
@info "Predicting"
predictedDataframe = KeypointsDetection.predict_to_dataframe(net, augmentedDataframe; needed_columns);
KeypointsDetection.show_image(predictedDataframe, 1)
readline()

# show the best, the worst and the barchart with mean error
sortedDataframe = KeypointsDetection.sort_by_error(predictedDataframe, augmentedDataframe)

# the worst
KeypointsDetection.show_image(sortedDataframe, 28185; goldDataset=augmentedDataframe)
readline()

# the best
KeypointsDetection.show_image(sortedDataframe, 100; goldDataset=augmentedDataframe)
readline()

# the average
KeypointsDetection.show_image(sortedDataframe, 10000; goldDataset=augmentedDataframe)
readline()

KeypointsDetection.plot_errors(sortedDataframe)
readline()

end