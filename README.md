# KeypointsDetection

The main aim of this project is to try to play with `VideoIO`, `GLMakie`, and `ObjectDetector`. I chose the dataset for [Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection/overview) from Kaggle. The image is captured by the webcam, then pretrained YOLO net is used to retrieve all the regions with faces. I train a network on the aforementioned Kaggle dataset, and then predict the location of facial keypoints on these regions.

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
- automatic download of YOLO model [ ]
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