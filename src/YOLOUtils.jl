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
end

function prepare_yolo_structs(image, model)
    imageratio = size(image,2) / size(image,1)
    modelratio = model.cfg[:width] / model.cfg[:height]

    if modelratio > imageratio
        h, w = size(image,1) .* (1, modelratio)
    else
        h, w = size(image,2) ./ (modelratio, 1)
    end

    x1i, y1i, x2i, y2i = [1, 2, 3, 4]

    DrawStruct(h, w, x1i, y1i, x2i, y2i)
end

unpack(s::DrawStruct) = s.h, s.w, s.x1i, s.y1i, s.x2i, s.y2i

function create_bounding_box(results, padding, s::DrawStruct)
    boxes = GeometryBasics.Polygon[]
    h, w, x1i, y1i, x2i, y2i = unpack(s)
    for i in 1:size(results,2)
        bbox = results[1:4, i] .- padding
        p1 = GeometryBasics.Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y1i]*h)+1)
        p2 = GeometryBasics.Point(round(Int, bbox[x2i]*w), round(Int, bbox[y1i]*h)+1)
        p3 = GeometryBasics.Point(round(Int, bbox[x1i]*w)+1, round(Int, bbox[y2i]*h))
        p4 = GeometryBasics.Point(round(Int, bbox[x2i]*w), round(Int, bbox[y2i]*h))
        push!(boxes,
            GeometryBasics.Polygon([
                p1, p2, p3, p4
            ])
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
    create_bounding_box(res, pad, s)
end
