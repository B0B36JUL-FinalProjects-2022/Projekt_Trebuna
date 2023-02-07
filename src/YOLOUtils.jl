using ObjectDetector
using GeometryBasics

function load_yolo_model()
    model = YOLO.yolo(
        "data/face-yolov3-tiny_41000.cfg",
        "data/face-yolov3-tiny_41000.weights"
    )
end

struct DrawStruct
    h::Float32
    w::Float32
    x1i::Int32
    y1i::Int32
    x2i::Int32
    y2i::Int32
    modelratio::Float32
    imgratio::Float32
end

function prepare_yolo_structs(img, model; transpose=true)
    h, w = ObjectDetector.getModelInputSize(model)
    h = size(img, 1)
    w = size(img, 2)
    modelratio, imgratio = 1, 1
    x1i, y1i, x2i, y2i = [1, 2, 3, 4]

    DrawStruct(h, w, x1i, y1i, x2i, y2i, modelratio, imgratio)
end

unpack(s::DrawStruct) = s.h, s.w, s.x1i, s.y1i, s.x2i, s.y2i, s.modelratio, s.imgratio

function init_dummy_bb(h, w)
    GeometryBasics.Polygon[
        GeometryBasics.Polygon(Point{2, Float32}.(
            [(0, 0), (w, 0), (w, h), (0, h), (0, 0)])
        )
    ]
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

function predict_bounding_box(image, model, s::DrawStruct;
    detectThresh=0.5,
    overlapThresh=0.8
)
    im, pad = prepareImage(image, model)
    batch = emptybatch(model)
    batch[:, :, :, 1] = im
    res = model(
        batch,
        detectThresh=detectThresh,
        overlapThresh=overlapThresh
    )
    create_bounding_box(res, pad, s), im
end
