module KeypointsDetection

include("Utils.jl")
include("LoadDataset.jl")
include("TrainNet.jl")
include("TrainUtils.jl")
include("PredictUtils.jl")
include("ImageUtils.jl")
include("Augmentation.jl")
include("ValidateUtils.jl")
include("YOLOUtils.jl")

import GLMakie
import VideoIO


function play_test_video()
    f = VideoIO.testvideo("annie_oakley")
    VideoIO.playvideo(f)
end

function play_webcam(model::YOLO.yolo, net::NetHolder)
    cam = VideoIO.opencamera()
    try
        img = read(cam)
        yolo_struct = prepare_yolo_structs(img, model)
        boxes, im = predict_bounding_box(img, model, yolo_struct)
        predict_from_bb(net, im, boxes)
        
        # --- observables can be interactively updated ---
        
        # create observable image for the input from the web-cam
        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        
        # create observable plot for the BoundingBox
        obs_plot = GLMakie.Observable(boxes)
        
        # ---
        
        # --- create a scene where everything is going to be displayed
        scene = GLMakie.Scene()
        
        # plot an image into the scene
        # GLMakie.Figure(; resolution=resolution)
        GLMakie.image!(scene, obs_img)
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
        while GLMakie.isopen(scene)
          img = read(cam)
          boxes, im = predict_bounding_box(img, model, yolo_struct)
          predict_from_bb(net, im, boxes)
          obs_plot[] = boxes
          obs_img[] = GLMakie.rotr90(img)
          sleep(1 / fps)

        end
    finally
        close(cam)
    end
end

end
