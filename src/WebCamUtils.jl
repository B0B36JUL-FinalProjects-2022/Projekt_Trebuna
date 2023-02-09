import .GLMakie
import .VideoIO
import .ObjectDetector
using GeometryBasics

export play_webcam

"""
For some reason GLMakie has inverted y-axis, e.g. all the image
count y == 0 in the top 
"""
function transform_predictions(w, h, boxes)
    new_boxes = GeometryBasics.Polygon[]
    for box in boxes
        for (i, point) in enumerate(box)
            box[i] = (point[1], h - point[2])
        end
        push!(new_boxes, 
            GeometryBasics.Polygon(
                GeometryBasics.Point{2, Float32}.(box)
            ))
    end
    new_boxes
end

"""
Predict the keypoints for each bounding box in `boxes`. Returns the keypoints without
the information to which boundingbox they belong. (It is then easy to just call scatter to display all of them)
"""
function get_keypoints(net::NetHolder, img, boxes)
    points = [(1, 1)]
    h = size(img, 1)
    grayscale_im = KeypointsDetection.to_grayscale(img)
    for bbox in boxes[2:end]
        cropped_grayscale_im, old_x, old_y, r, p11, p21 = KeypointsDetection.crop_out_face(grayscale_im, bbox)
        preds = KeypointsDetection.predict(net, cropped_grayscale_im; withgpu=true)
        preds = collect(zip(preds[1:2:end], 96 .- preds[2:2:end]))
        preds = KeypointsDetection.preds_to_full(preds, old_x, old_y, r, p11, p21)
        points = vcat(points, preds)
    end
    points = GeometryBasics.Point{2, Float32}.([(x, y) for (x, y) in points])
    points
end

"""
According to the output of the system web-camera, display the bounding box predicted by the
`model::Yolo.yolo` and the facial keypoints predicted by the `net::NetHolder`.
"""
function play_webcam(
    model::YOLO.yolo,
    net::NetHolder;
    net_preds_per_second::T=5
) where T <: Integer
    cam = VideoIO.opencamera()
    try
        @info "First camera read"
        img = read(cam)

        @info "First bounding box prediction"
        yolo_struct = prepare_yolo_structs(img, model)
        boxes = predict_bounding_box(img, model, yolo_struct)
        boxes = transform_predictions(yolo_struct.im_w, yolo_struct.im_h, boxes)
        points = get_keypoints(net, img, boxes)
        
        # --- observables can be interactively updated ---
        
        # create observable image for the input from the web-cam
        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        
        # create observable plot for the BoundingBox
        obs_plot = GLMakie.Observable(boxes)

        # create observable for facial keypoints
        obs_keypoints = GLMakie.Observable(points)
        
        # --- create a scene where everything is going to be displayed
        scene = GLMakie.Scene()
        
        # plot an image into the scene
        @info "First display of camera output and predictions"
        GLMakie.image!(scene, obs_img)
        GLMakie.poly!(
            scene,
            obs_plot,
            color=:transparent,
            strokecolor=:red,
            strokewidth=2,
            overdraw=true
        )

        GLMakie.scatter!(
            scene,
            obs_keypoints,
            color=:yellow
        )
            
        display(scene)
        fps = VideoIO.framerate(cam)
        @info "Start of while loop"
        last_pred = -1
        while GLMakie.isopen(scene)
            img = read(cam)
            obs_img[] = GLMakie.rotr90(img)
            if (last_pred == -1) || ((time() - last_pred) > (1 / net_preds_per_second))
                last_pred = time()
            else
                sleep(1 / fps)
                continue
            end
            boxes = predict_bounding_box(img, model, yolo_struct)
            boxes = transform_predictions(yolo_struct.im_w, yolo_struct.im_h, boxes)
            points = get_keypoints(net, img, boxes)
            if length(points) > 0
                obs_keypoints[] = points
            end
            obs_plot[] = boxes
            sleep(1 / fps)
        end
    finally
        close(cam)
    end
end