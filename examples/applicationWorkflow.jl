module ApplicationWorkflow
using Revise
using KeypointsDetection
using Plots
using FileIO
using ObjectDetector

function show_image(img::AbstractArray)
    plt = Plots.plot(img)
    gui(plt)
end

function show_image_with_bbox(img::AbstractArray, bbox::AbstractArray)
    plt = Plots.plot(img)
    Plots.plot!(bbox, label="Bounding box")
    gui(plt)
end

function show_gpu_image_scatter(img, sc)
    im_show = img |> cpu
    im_show = channels_to_rgb(im_show)
    plt = Plots.plot(im_show)
    for (i,s) in enumerate(sc)
        Plots.scatter!([s], label="$i")
    end
    gui(plt)
end

function show_gpu_image(im)
    im_show = im |> cpu
    im_show = channels_to_rgb(im_show)
    show_image(im_show)
end

@info "This module displays the traditional workflow of an app"
@info "1. Load an image taken on the webcam of my pc"
img = load("examples/images/2023-02-08-154201.jpg");
show_image(img);
readline()

@info "2. Load yolo model for detection of face"
model = load_yolo_model()

@info "3. The image needs to be transformed to fit into 416x416x3 size needed by the yolo model"
transformed_img = get_prepared_image(img, model);
show_image(transformed_img);
readline()

@info "4. The face bounding box for the transformed image is predicted"
yolo_struct = prepare_yolo_structs(img, model);
res, pad, im = yolo_predict(img, model);
bbox_transformed = res_to_bounding_box(res[:, 1], yolo_struct.m_h, yolo_struct.m_w)
show_image_with_bbox(transformed_img, bbox_transformed)
readline()

@info "5. Transform the bounding box to the coordinates of the original image"
bbox = res_to_bounding_box(res[:, 1], yolo_struct.im_h, yolo_struct.im_w, pad)
show_image_with_bbox(img, bbox)
readline()

@info "6. Models for keypoint detection were trained on grayscale images"
grayscale_im = to_grayscale(img);
show_gpu_image(grayscale_im)
readline()

@info "7. Crop out the face from the grayscale image"
cropped_grayscale_im, old_x, old_y, r, p11, p21 = crop_out_face(grayscale_im, bbox);
show_gpu_image(cropped_grayscale_im);
readline()

@info "8. Load pretrained model for keypoints detection"
model_path="examples/models/traits_model.bson";
needed_columns=columns_basic_traits;
net = define_net_lenet_dropout(;n_outputs=length(needed_columns), dropout_rate=0.2);
load_net(net, model_path)

@info "9. Predict keypoints on the cropped out face"
preds = predict(net, cropped_grayscale_im; withgpu=true)
preds = collect(zip(preds[1:2:end], preds[2:2:end]))
show_gpu_image_scatter(cropped_grayscale_im, preds)
readline()

@info "10. Rescale back to the original image"
new_preds = preds_to_full(preds, old_x, old_y, r, p11, p21)
show_gpu_image_scatter(grayscale_im, new_preds)
readline()


end