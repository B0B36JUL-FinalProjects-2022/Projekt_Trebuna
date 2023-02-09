module Webcam

using Revise
using KeypointsDetection
using GLMakie
using VideoIO
using ObjectDetector

# load facial-keypoints net
model_path="examples/models/traits_model_full_2.bson"
#needed_columns=columns_basic_traits
#net = define_net_lenet_dropout(dropout_rate=0.2, n_outputs=length(needed_columns))
net = define_net_lenet_deeper_dropout(dropout_rate=0.2)
load_net(net, model_path)

# load yolo net
model = load_yolo_model()

# start webcam
play_webcam(model, net)

end