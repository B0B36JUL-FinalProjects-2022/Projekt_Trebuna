module VideoExample
using Revise
using KeypointsDetection

trainDataframe = KeypointsDetection.load_train_dataframe();
X, y = KeypointsDetection.create_train_dataset_full(trainDataframe);
net = KeypointsDetection.define_net_simple_feedforward()
train_losses_steps, valid_losses = KeypointsDetection.train_gpu_net!(net, X, y);
KeypointsDetection.plot_losses(train_losses_steps, valid_losses)
predictedDataframe = KeypointsDetection.predict_to_dataframe(net, trainDataframe);
KeypointsDetection.show_image(predictedDataframe, 1)


end # VideoExample