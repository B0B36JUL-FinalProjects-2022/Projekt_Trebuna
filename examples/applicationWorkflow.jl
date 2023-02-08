module ApplicationWorkflow
using Revise
using KeypointsDetection
using Plots
using FileIO
using ObjectDetector

function show_image(img::AbstractArray)
    plt = plot(img)
    gui(plt)
end

function show_image_with_bbox(img::AbstractArray, bbox::AbstractArray)
    plt = plot(img)
    plot!(bbox, label="Bounding box")
    gui(plt)
end

function show_gpu_image_scatter(img, sc)
    im_show = img |> cpu
    im_show = KeypointsDetection.channels_to_rgb(im_show)
    plt = plot(im_show)
    for (i,s) in enumerate(sc)
        scatter!([s], label="$i")
    end
    gui(plt)
end

function show_gpu_image(im)
    im_show = im |> cpu
    im_show = KeypointsDetection.channels_to_rgb(im_show)
    show_image(im_show)
end

@info "This module displays the traditional workflow of an app"
@info "1. Load an image taken on the webcam of my pc"
img = load("examples/images/2023-02-08-154201.jpg");
show_image(img);
readline()

@info "2. Load yolo model for detection of face"
model = KeypointsDetection.load_yolo_model()

@info "3. The image needs to be transformed to fit into 416x416x3 size needed by the yolo model"
transformed_img = KeypointsDetection.get_prepared_image(img, model);
show_image(transformed_img);
readline()

@info "4. The face bounding box for the transformed image is predicted"
yolo_struct = KeypointsDetection.prepare_yolo_structs(img, model);
res, pad, im = KeypointsDetection.yolo_predict(img, model);
bbox_transformed = KeypointsDetection.res_to_bounding_box(res[:, 1], yolo_struct.m_h, yolo_struct.m_w)
show_image_with_bbox(transformed_img, bbox_transformed)
readline()

@info "5. Transform the bounding box to the coordinates of the original image"
bbox = KeypointsDetection.res_to_bounding_box(res[:, 1], yolo_struct.im_h, yolo_struct.im_w, pad)
show_image_with_bbox(img, bbox)
readline()

@info "6. Models for keypoint detection were trained on grayscale images"
grayscale_im = KeypointsDetection.to_grayscale(img);
show_gpu_image(grayscale_im)
readline()

@info "7. Crop out the face from the grayscale image"
cropped_grayscale_im = KeypointsDetection.crop_out_face(grayscale_im, bbox);
show_gpu_image(cropped_grayscale_im);
readline()

@info "8. Load pretrained model for keypoints detection"
model_path="examples/models/traits_model.bson";
needed_columns=KeypointsDetection.columns_basic_traits;
net = KeypointsDetection.define_net_lenet_dropout(;n_outputs=length(needed_columns), dropout_rate=0.2);
KeypointsDetection.load_net(net, model_path)

@info "9. Predict keypoints on the cropped out face"
preds = KeypointsDetection.predict(net, cropped_grayscale_im; withgpu=false)
preds = collect(zip(preds[1:2:end], preds[2:2:end]))
show_gpu_image_scatter(cropped_grayscale_im, preds)
readline()


end