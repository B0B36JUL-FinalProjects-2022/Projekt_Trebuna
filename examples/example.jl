module VideoExample
using Revise
using KeypointsDetection

trainDataframe = KeypointsDetection.load_train_dataframe();

# showcase of different augmentations
horizontallyFlippedDataframe = KeypointsDetection.horizontalFlip(trainDataframe);
N = 5
rotationDataframe = KeypointsDetection.rotation(trainDataframe, pi * 1 / N);
negRotationDataframe = KeypointsDetection.rotation(trainDataframe, -pi * 1 / N);
KeypointsDetection.show_image_augmented(
    [
        trainDataframe,
        horizontallyFlippedDataframe,
        rotationDataframe,
        negRotationDataframe
    ], 
    [
        "Original",
        "Horizontal Flip",
        "Rotation pi/$N",
        "Rotation - pi/$N"
    ],
    N
)


X, y = KeypointsDetection.create_train_dataset_full(trainDataframe);


net = KeypointsDetection.define_net_simple_feedforward()
train_losses_steps, valid_losses = KeypointsDetection.train_gpu_net!(net, X, y);
KeypointsDetection.plot_losses(train_losses_steps, valid_losses)
predictedDataframe = KeypointsDetection.predict_to_dataframe(net, trainDataframe);
KeypointsDetection.show_image(predictedDataframe, 1)


end # VideoExample