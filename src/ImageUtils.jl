function plot_keypoints(dataset::DataFrame, index)
    eye_group = [
        "eye_inner_corner",
        "eye_center",
        "eye_outer_corner"
    ]
    eyebrow_group = [
        "eyebrow_inner_end",
        "eyebrow_outer_end"
    ]
    mouth_left_right_group = [
        "mouth_left_corner",
        "mouth_right_corner"
    ]
    mouth_top_down_group = [
        "mouth_center_top_lip",
        "mouth_center_bottom_lip"
    ]
    nose_group = ["nose_tip"]
    prefix(group, what) = [what * name for name in group]
    prefix_left(group) = prefix(group, "left_")
    prefix_right(group) = prefix(group, "right_")
    groups = [ 
        (prefix_left(eye_group), "eye_left"),
        (prefix_right(eye_group), "eye_right"),
        (prefix_left(eyebrow_group), "eyebrow_left"),
        (prefix_right(eyebrow_group), "eyebrow_right"),
        (nose_group, "nose"),
        (mouth_left_right_group, "mouth_left_right"),
        (mouth_top_down_group, "mouth_top_bottom")
    ]
    for (group, name) in groups
        x = []
        y = []
        for (dim, arr) in zip(["_x", "_y"], [x, y])
            append!(arr, [
                dataset[index, prefix * dim] for prefix in group
            ])
        end
        plot!(x, y, m=:o, label=name)
    end
end

"""
Show the image and important keypoints.
"""
function show_image(dataset::DataFrame, index::Integer)
    image = dataset[index, :Image]
    plt = plot(image)
    plot_keypoints(dataset, index)
    plt
end