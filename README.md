# KeypointsDetection

### TODOs

- webcam connection [x]
- a model which detects everything at once [x]
- a model detecting just the basic traits [x]
- using `ObjectDetector.jl` to extract  faces from the webcam image [x]
- resizing and adjusting extracted faces to fit into trained model [ ]
- automatic download of YOLO model [ ]
- faster start time using precompilation [ ]
- ResNet for traits detection [ ]
- docstrings [ ]
- tests [ ]

### References

- [tutorial on facial keypoints recognition](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
- [pretrained weights for yolo for facial detection](https://github.com/lthquy/Yolov3-tiny-Face-weights)
- [how to define yolo](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

### Issues

#### `GLMakie`
- it was needed to create an environment variable: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` because `julia` kept using the wrong version of `libstdc++`  