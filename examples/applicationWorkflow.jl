module ApplicationWorkflow
using Revise
using KeypointsDetection
using Plots
using FileIO

function show_image(img::AbstractArray)
    plt = plot(image)
    gui(plt)
end

@info "This module displays the traditional workflow of an app"
@info "At first, let's load an image taken on the webcam of my pc"

img = load("examples/images/2023-02-07-103125.jpg");
show_image(img);

@info "2. Load yolo model for detection of face"
model = KeypointsDetection.load_yolo_model()

@info "3. The image needs to be transformed to fit into 416x416x3 size needed by the yolo model"
transformed_img = KeypointsDetection.show_prepared_image(img, model);
show_image(transformed_img);

end