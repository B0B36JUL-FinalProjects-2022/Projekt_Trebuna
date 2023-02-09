using .ObjectDetector
using GeometryBasics
using Images

export load_yolo_model, predict_bounding_box, yolo_predict, res_to_bounding_box, get_prepared_image, prepare_yolo_structs, channels_to_rgb

"""
Load pretrained weights and configuration from `joinpath(model_dir, cfg_name)`
and `joinpath(model_dir, weights_name)` create an instance of `YOLO` model based on the
loaded values.
"""
function load_yolo_model(;
    model_dir::String="data",
    cfg_name::String="face-yolov3-tiny_41000.cfg",
    weights_name::String="face-yolov3-tiny_41000.weights"
)
    model = YOLO.yolo(
        joinpath(model_dir, cfg_name),
        joinpath(model_dir, weights_name)
    )
end

struct DrawStruct
    im_h::Float32
    im_w::Float32
    m_h::Float32
    m_w::Float32
end

"""
Store the original image size as well as the size that the model expects.
"""
function prepare_yolo_structs(img, model; transpose=true)
    m_h, m_w = ObjectDetector.getModelInputSize(model)
    im_h = size(img, 1)
    im_w = size(img, 2)

    DrawStruct(im_h, im_w, m_h, m_w)
end

unpack(s::DrawStruct) = s.h, s.w, s.x1i, s.y1i, s.x2i, s.y2i, s.modelratio, s.imgratio

function init_dummy_bb(h, w)
    [[(0, 0), (w, 0), (w, h), (0, h), (0, 0)]]
end

"""
Transform the output of the yolo model into the coordinate set of the model input, e.g. to `(w, h)` image.

Args:
- results: outputs of the yolo model
- h: model input height
- w: model input width
"""
function res_to_bounding_box(results::Vector{Float32}, h::Float32, w::Float32)
    bbox = results[1:4]
    p1 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[2] * h) + 1)
    p2 = (round(Int, bbox[3] * w),      round(Int, bbox[2] * h) + 1)
    p3 = (round(Int, bbox[3] * w),      round(Int, bbox[4] * h))
    p4 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[4] * h))
    return [p1, p2, p3, p4, p1]
end

"""
Transform the output of the yolo model into the coordinate set of the original image, e.g. before padding,
and scale shift.

Args:
- results: outputs of the yolo model
- h: original height
- w: original width
- padding: the padding needed to make the orignal image shape fit into the model shape
"""
function res_to_bounding_box(results::Vector{Float32}, h::Float32, w::Float32, padding::AbstractArray{Float64})    
    bbox = results[1:4] .- padding
    h *= (w / h)
    p1 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[2] * h) + 1)
    p2 = (round(Int, bbox[3] * w),      round(Int, bbox[2] * h) + 1)
    p3 = (round(Int, bbox[3] * w),      round(Int, bbox[4] * h))
    p4 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[4] * h))
    return [p1, p2, p3, p4, p1]
end

"""
Transform `(w, h, c)` shaped array of floats into `(w, h)` array of `RGB`
"""
function channels_to_rgb(im)
    im_rgb_unwinded = [RGB(im[i, j, :]...) for j in range(1, size(im, 2)) for i in range(1, size(im, 1))]
    im_rgb = reshape(im_rgb_unwinded, size(im, 1), size(im, 2))'
end

function get_prepared_image(image, model)
    im, pad = prepareImage(image, model)
    im = im |> cpu
    channels_to_rgb(im)
end

"""
Predict the coordinates of all the facial bounding boxes in the image. This method
automatically creates batch, pads the image, predicts the values for the batch and
returns the raw numbers from the model. 

E.g. to draw the bounding boxes returned by this method it is advised to call
`predict_bounding_box` which does transform the coordinates automagically.

Args:
- image
- model: the yolo model loaded by `load_yolo_model`
"""
function yolo_predict(image, model; detectThresh=0.5, overlapThresh=0.8)
    im, pad = prepareImage(image, model)
    batch = emptybatch(model)
    batch[:, :, :, 1] = im
    res = model(
        batch,
        detectThresh=detectThresh,
        overlapThresh=overlapThresh
    )
    res, pad, im
end


"""
Predict the coordinates of all the facial bounding boxes in the image.

Args:
- image
- model: the yolo model loaded by `load_yolo_model`
- `s::DrawStruct`: the structure holding the original image size as well as the model
    input size, it is used to transform the predictions of the yolo model into the
    original coordinate space
"""
function predict_bounding_box(image, model, s::DrawStruct;
    detectThresh=0.5,
    overlapThresh=0.8
)
    boxes = init_dummy_bb(s.im_h, s.im_w)
    res, pad, im = yolo_predict(image, model; detectThresh, overlapThresh)
    for i in range(1, size(res, 2))  
        box = res_to_bounding_box(res[:, i], s.im_h, s.im_w, pad)
        push!(boxes, box)
    end
    boxes
end
