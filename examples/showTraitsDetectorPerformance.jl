module TraitsDetectorPerformance
using KeypointsDetection
using DataFrames
using Plots

@info "Loading Dataset"
trainDataframe = load_train_dataframe();
augmentedDataframe = augment_dataframe(trainDataframe);

model_path="examples/models/traits_model_full.bson"
@info "Loading Trained Model $(model_path)"
#needed_columns=columns_basic_traits
net = define_net_lenet_dropout(;dropout_rate=0.2) #n_outputs=length(needed_columns))
load_net(net, model_path)

# show some prediction
@info "Predicting"
predictedDataframe = predict_to_dataframe(net, augmentedDataframe)#; needed_columns);
@info "Prediction on random photo from the dataset"
show_image_and_keypoints(predictedDataframe, 1)
readline()

# show the best, the worst and the barchart with mean error
sortedDataframe = sort_by_error(predictedDataframe, augmentedDataframe)

@info "Worst prediction"
show_image_with_gold(sortedDataframe, 28185; goldDataframe=augmentedDataframe)
readline()

@info "Best prediction"
show_image_with_gold(sortedDataframe, 100; goldDataframe=augmentedDataframe)
readline()

@info "Average prediction"
show_image_with_gold(sortedDataframe, 10000; goldDataframe=augmentedDataframe)
readline()

@info "Errors"
show_errors(sortedDataframe)
readline()

end