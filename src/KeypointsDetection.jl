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

function play_webcam(model)
    cam = VideoIO.opencamera()
    fps = VideoIO.framerate(cam)
    try
        img = read(cam)
        yolo_struct = prepare_yolo_structs(img, model)
        boxes = predict_bounding_box(img, model, yolo_struct)

        # --- observables can be interactively updated ---

        # create observable image for the input from the web-cam
        obs_img = GLMakie.Observable(GLMakie.rotr90(img))

        # create observable plot for the BoundingBox
        obs_plot = GLMakie.Observable(boxes)

        # ---

        # --- create a scene where everything is going to be displayed
        resolution = reverse(size(img))
        scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=resolution)
        
        # plot an image into the scene
        GLMakie.image!(scene, obs_img)
        GLMakie.poly!(scene, obs_plot)
      
        display(scene)
        
        while GLMakie.isopen(scene)
          img = read(cam)
          obs_plot[] = predict_bounding_box(img, model, yolo_struct)
          obs_img[] = GLMakie.rotr90(img)
          sleep(1 / fps)
        end
    finally
        close(cam)
    end
end

# Write your package code here.
function something_better()
    println("better")
end

end
