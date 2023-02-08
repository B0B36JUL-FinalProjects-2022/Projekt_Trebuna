module Webcam

using Revise
using KeypointsDetection

# load facial-keypoints net
model_path="examples/models/traits_model.bson"
needed_columns=KeypointsDetection.columns_basic_traits
net = KeypointsDetection.define_net_lenet_dropout(n_outputs=length(needed_columns), dropout_rate=0.2)
KeypointsDetection.load_net(net, model_path)

# load yolo net
model = KeypointsDetection.load_yolo_model()

# start webcam
KeypointsDetection.play_webcam(model, net)


end