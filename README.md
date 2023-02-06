# KeypointsDetection

### TODOs

- webcam connection [x]
- a model which detects everything at once [x]
- a model detecting just the basic traits [x]
- using `ObjectDetector.jl` to extract  faces from the webcam image [ ]
- resizing and adjusting extracted faces to fit into trained model [ ]
- ResNet for traits detection [ ]
- docstrings [ ]
- tests [ ]


### Issues

#### `GLMakie`
- it was needed to create an environment variable: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` because `julia` kept using the wrong version of `libstdc++`  