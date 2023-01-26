module VideoExample
using Revise
using KeypointsDetection

export yolo

function yolo()
    println("Hello")
    KeypointsDetection.play_test_video()
end

end # VideoExample