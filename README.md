# KeypointsDetection

The project is composed of two parts:
1. Training and Evaluation of the model for [Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection/overview) from Kaggle.
2. Proof of concept implementation of a pipeline for detecting facial keypoints on image-stream captured by a pc webcam.

## [Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection/overview)

The package exports several utility functions for training of the models on the challenge.

### Data Loading
- `load_train_dataframe`, `augment_dataframe`, `create_train_dataset`

### Training and Evaluation
- model definitions: `define_net_simple_feedforward`, `define_net_lenet`, `define_net_lenet_dropout`
- training: `train_net!`, `train_gpu_net!`, `show_losses`
- evaluation: `predict_to_dataframe`, `show_image_and_keypoints`, `show_image_with_gold`, `show_errors`

### Results

I was able to replicate the performances reported by the official tutorial from kaggle ([link](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)). The models can be trained by scripts [`examples/trainTraitsDetector.jl`](examples/trainTraitsDetector.jl) for the basic traits (e.g. detect just eyes, nose, and the middle of mouth).

#### Example outputs
- basic traits:
![image_basic_traits](readme_files/basic_traits.jpg)

## Proof of Concept WebCam Pipeline

The main aim of this part is to try to play with `VideoIO`, `GLMakie`, and `ObjectDetector`. The pipeline 

A demonstration of the pipeline can be found in [examples/applicationWorkflow.jl](examples/applicationWorkflow.jl), a working proof of concept is in [examples/webcam.jl](examples/webcam.jl). The `gif` image below shows the output of the `webcam` script.

![webcam gif](readme_files/webcam.gif)

### TODOs

- webcam connection [x]
- a model which detects everything at once [x]
- a model detecting just the basic traits [x]
- using `ObjectDetector.jl` to extract  faces from the webcam image [x]
- resizing and adjusting extracted faces to fit into trained model [x]
- add `@require` macros to reduce load time of packages [x]
- tests [ ]
    - yolo change of resolution test - 4 different images, bounding boxes should be at the right place after the transformation [ ]
    - `channels_to_rgb`
- automatic download of YOLO model [-]
- faster start time using precompilation [-]
    - won't do as I do not want to spend time on sysimages
- ResNet for traits detection [ ]
- docstrings [ ]

### References

- [tutorial on facial keypoints recognition](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
- [pretrained weights for yolo for facial detection](https://github.com/lthquy/Yolov3-tiny-Face-weights)
- [how to define yolo](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

### Issues

#### `GLMakie`
- it was needed to create an environment variable: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` because `julia` kept using the wrong version of `libstdc++`  