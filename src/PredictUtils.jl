using GeometryBasics
using NNlib

export to_grayscale, crop_out_face, preds_to_full

"""
Transform the image to the grayscale, adjusting brightness as needed.
REFERENCE: https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
"""
function to_grayscale(img; brightness::Float64=2.0)
    img = permutedims(channelview(img), (3, 2, 1))
    reshape(
        brightness .* ( (1/3 .* img[:, :, 1]) .+ (img[:, :, 2] .* 1/3) .+ (img[:, :, 3] .* 1/3) ),
        (size(img, 1), size(img, 2), 1, 1)
    )
end

"""
If any side of the image is bigger than 96 pixels, then the image is transformed
in such a way that the larger side is exactly 96 pixels long, while preserving
the aspect ratio.
"""
function shrink_biggest_side_to_96_pixels(img)
    ratio = 1
    if size(img, 1) >= size(img, 2)
        if size(img, 1) > 96
            new_width=96
            ratio = 96 / size(img, 1)
            new_height=floor(Int64, size(img, 2) * ratio)
            img = imresize(img, new_width, new_height)
        end
    elseif size(img, 1) < size(img, 2)
        if size(img, 2) > 96
            ratio = 96 / size(img, 2)
            new_height=96
            new_width=floor(Int64, size(img, 1) * ratio)
            img = imresize(img, new_width, new_height)
        end
    end
    img, ratio
end

"""
Pad image with zeros in such a way that the new shape of `img`
is (96,96)
"""
function pad_image_to_96_pixels(img)
    p11, p12, p21, p22 = 0, 0, 0, 0
    p1 = 96 - size(img, 1)
    p2 = 96 - size(img, 2)
    if p1 > 0 && p2 > 0
        p11 = floor(Int, p1 / 2)
        p12 = Int(p1 - p11)
        p21 = floor(Int, p2 / 2)
        p22 = Int(p2 - p21)
    else
        if p1 > 0
            p11 = floor(Int, p1 / 2)
            p12 = Int(p1 - p11)
        elseif p2 > 0
            p21 = floor(Int, p2 / 2)
            p22 = Int(p2 - p21)
        end
    end
    img = NNlib.pad_constant(img, (p11, p12, p21, p22), 0)
    img, p11, p21
end

function crop_out_face(img, boundingbox::Vector{Tuple{Int64, Int64}})
    crop_out_face(img, GeometryBasics.Polygon(Point{2, Float32}.(boundingbox)))
end

"""
The region defined by the `boundingbox` is cropped out of the image, and then reshaped
and padded in such a way that the resulting returned image is of shape (96, 96)

Returns:
- `cropped_out_face`
- `old_x`, `old_y`, `ratio`, `padding_from_left`, `padding_from_top` the values needed to
transform the predictions to the original image
"""
function crop_out_face(img_orig, boundingbox::GeometryBasics.Polygon)
    points_x = []
    points_y = []
    for line in boundingbox.exterior.points
        push!(points_x, line.points[1][1])
        push!(points_y, line.points[1][2])
    end
    min_x = max(Int32(minimum(points_x)), 1)
    max_x = min(Int32(maximum(points_x)), size(img_orig, 1))
    min_y = max(Int32(minimum(points_y)), 1)
    max_y = min(Int32(maximum(points_y)), size(img_orig, 2))

    img = img_orig[min_x:max_x, min_y:max_y]
    old_x = min_x
    old_y = min_y

    img, ratio = shrink_biggest_side_to_96_pixels(img)
    img, p11, p21 = pad_image_to_96_pixels(img)
    
    convert(Matrix{Float32}, img), old_x, old_y, ratio, p11, p21
end

"""
Take predictions on the cropped out image (the output of `crop_out_face` method) and
transform them back to the full size image. All the remaining arguments of this method
are the other outputs of `crop_out_face`

Example:
```julia
cropped_grayscale_im, old_x, old_y, r, p11, p21 = crop_out_face(grayscale_im, bbox);
preds = predict(net, cropped_grayscale_im; withgpu=true)
# IMPORTANT: the predictions need to be transformed to point format!
# e.g. like this
preds = collect(zip(preds[1:2:end], preds[2:2:end]))
new_preds = preds_to_full(preds, old_x, old_y, r, p11, p21)
```

Returns:
- predictions transformed to the original image
"""
function preds_to_full(preds, old_x, old_y, ratio, p11, p21)
    new_preds = []
    for p in preds
        x = p[1] - p11
        y = p[2] - p21
        push!(new_preds, (x / ratio + old_x, y / ratio + old_y))
    end
    new_preds
end

