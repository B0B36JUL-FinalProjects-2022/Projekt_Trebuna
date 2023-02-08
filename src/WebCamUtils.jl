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

function play_webcam(model::YOLO.yolo, net::NetHolder)
    cam = VideoIO.opencamera()
    try
        @info "First camera read"
        img = read(cam)

        @info "First bounding box prediction"
        yolo_struct = prepare_yolo_structs(img, model)
        boxes = predict_bounding_box(img, model, yolo_struct)
        boxes = transform_predictions(yolo_struct.im_w, yolo_struct.im_h, boxes)
        
        # --- observables can be interactively updated ---
        
        # create observable image for the input from the web-cam
        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        
        # create observable plot for the BoundingBox
        obs_plot = GLMakie.Observable(boxes)
        
        # --- create a scene where everything is going to be displayed
        scene = GLMakie.Scene()
        
        # plot an image into the scene
        @info "First display of camera output and predictions"
        GLMakie.image!(scene, obs_img)
        @show boxes
        GLMakie.poly!(
            scene,
            obs_plot,
            color=:transparent,
            strokecolor=:red,
            strokewidth=2,
            overdraw=true
        )
            
        display(scene)
        fps = VideoIO.framerate(cam)
        @info "Start of while loop"
        while GLMakie.isopen(scene)
            img = read(cam)
            boxes = predict_bounding_box(img, model, yolo_struct)
            boxes = transform_predictions(yolo_struct.im_w, yolo_struct.im_h, boxes)
            # predict_from_bb(net, im, boxes)
            obs_plot[] = boxes
            obs_img[] = GLMakie.rotr90(img)
            sleep(1 / fps)
        end
    finally
        close(cam)
    end
end