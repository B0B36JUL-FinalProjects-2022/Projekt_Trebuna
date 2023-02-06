module KeypointsDetection

include("Utils.jl")
include("LoadDataset.jl")
include("TrainNet.jl")
include("TrainUtils.jl")
include("PredictUtils.jl")
include("ImageUtils.jl")
include("Augmentation.jl")
include("ValidateUtils.jl")

import GLMakie
import VideoIO


function play_test_video()
    f = VideoIO.testvideo("annie_oakley")
    VideoIO.playvideo(f)
end

function play_webcam()
    cam = VideoIO.opencamera()
    fps = VideoIO.framerate(cam)
    try
        img = read(cam)
        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=reverse(size(img)))
        GLMakie.image!(scene, obs_img)
      
        display(scene)
      
        fps = VideoIO.framerate(cam)
        while GLMakie.isopen(scene)
          img = read(cam)
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
