module AugmentationShowcase

using Revise
using KeypointsDetection
using DataFrames
using Plots

@info "Load Train Dataframe"
trainDataframe = load_train_dataframe()

@info "Augment Train Dataframe"
horizontallyFlipped = horizontal_flip(trainDataframe)
negRotated = rotation(trainDataframe, -pi * 1 / 5)
rotated = rotation(trainDataframe, pi * 1 / 5)

@info "Display"
KeypointsDetection.show_image_augmented(
    [
        trainDataframe,
        horizontallyFlipped,
        negRotated,
        rotated
    ],
    [
        "Original",
        "Horizontal Flip",
        "Rotation -pi/5",
        "Rotation pi/5"
    ],
    15
)
readline()

end