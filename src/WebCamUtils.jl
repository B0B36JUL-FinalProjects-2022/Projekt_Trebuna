import .GLMakie
import .VideoIO
import .ObjectDetector
using GeometryBasics

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

function get_keypoints(net::NetHolder, img, boxes)
    points = [(1, 1)]
    h = size(img, 1)
    grayscale_im = KeypointsDetection.to_grayscale(img)
    for bbox in boxes[2:end]
        cropped_grayscale_im, old_x, old_y, r, p11, p21 = KeypointsDetection.crop_out_face(grayscale_im, bbox)
        preds = KeypointsDetection.predict(net, cropped_grayscale_im; withgpu=true)
        preds = preds |> cpu
        preds = collect(zip(preds[1:2:end], 96 .- preds[2:2:end]))
        preds = KeypointsDetection.preds_to_full(preds, old_x, old_y, r, p11, p21)
        points = vcat(points, preds)
    end
    points = GeometryBasics.Point{2, Float32}.([(x, y) for (x, y) in points])
    points
end

function play_webcam(model::YOLO.yolo, net::NetHolder)
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
        while GLMakie.isopen(scene)
            img = read(cam)
            boxes = predict_bounding_box(img, model, yolo_struct)
            boxes = transform_predictions(yolo_struct.im_w, yolo_struct.im_h, boxes)
            points = get_keypoints(net, img, boxes)
            if length(points) > 0
                obs_keypoints[] = points
            end
            obs_plot[] = boxes
            obs_img[] = GLMakie.rotr90(img)
            sleep(1 / fps)
        end
    finally
        close(cam)
    end
end