using .ObjectDetector
using GeometryBasics
using Images

function load_yolo_model(;model_dir::String="data")
    model = YOLO.yolo(
        joinpath(model_dir,"face-yolov3-tiny_41000.cfg"),
        joinpath(model_dir, "face-yolov3-tiny_41000.weights")
    )
end

struct DrawStruct
    im_h::Float32
    im_w::Float32
    m_h::Float32
    m_w::Float32
end

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

function create_bounding_box(results, padding, s::DrawStruct)
    h, w, x1i, y1i, x2i, y2i, _, imgratio = unpack(s)
    boxes = init_dummy_bb(h, w)
    length(results) == 0 && return boxes
    for i in 1:size(results,2)
        bbox = results[1:4, i]
        p1 = (round(Int, bbox[x1i]*w)+1, round(Int,  h - (bbox[y1i] - padding[2] / 2)*h)+1)
        p2 = (round(Int, bbox[x2i]*w), round(Int, h - (bbox[y1i] - padding[2] / 2)*h)+1)
        p3 = (round(Int, bbox[x1i]*w)+1, round(Int, h - (bbox[y2i] + padding[4] / 2)*h))
        p4 = (round(Int, bbox[x2i]*w), round(Int, h - (bbox[y2i] + padding[4] / 2)*h))
        push!(boxes,
            GeometryBasics.Polygon(Point{2, Float32}.([p1, p2, p4, p3, p1]))
        )
    end
    boxes
end

function res_to_bounding_box(results::Vector{Float32}, h::Float32, w::Float32)
    bbox = results[1:4]
    p1 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[2] * h) + 1)
    p2 = (round(Int, bbox[3] * w),      round(Int, bbox[2] * h) + 1)
    p3 = (round(Int, bbox[3] * w),      round(Int, bbox[4] * h))
    p4 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[4] * h))
    return [p1, p2, p3, p4, p1]
end

function res_to_bounding_box(results::Vector{Float32}, h::Float32, w::Float32, padding::AbstractArray{Float64})    
    bbox = results[1:4] .- padding
    h *= (w / h)
    p1 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[2] * h) + 1)
    p2 = (round(Int, bbox[3] * w),      round(Int, bbox[2] * h) + 1)
    p3 = (round(Int, bbox[3] * w),      round(Int, bbox[4] * h))
    p4 = (round(Int, bbox[1] * w) + 1,  round(Int, bbox[4] * h))
    return [p1, p2, p3, p4, p1]
end

function channels_to_rgb(im)
    im_rgb_unwinded = [RGB(im[i, j, :]...) for j in range(1, size(im, 2)) for i in range(1, size(im, 1))]
    im_rgb = reshape(im_rgb_unwinded, size(im, 1), size(im, 2))'
end

function get_prepared_image(image, model)
    im, pad = prepareImage(image, model)
    im = im |> cpu
    channels_to_rgb(im)
end

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

function predict_bounding_box(image, model, s::DrawStruct;
    detectThresh=0.5,
    overlapThresh=0.8
)
    boxes = init_dummy_bb(s.im_h, s.im_w)
    res, pad, im = yolo_predict(image, model; detectThresh, overlapThresh)
    for i in range(1, size(res, 2))  
        box = KeypointsDetection.res_to_bounding_box(res[:, i], s.im_h, s.im_w, pad)
        push!(boxes, box)
    end
    boxes
end
