using Images
export channels_to_rgb

"""
Measures the time needed to run the `method` argument, returns the time
"""
function measure_time(method)
    s = time()
    method()
    e = time()
    e - s
end

"""
Transform `(w, h, c)` shaped array of floats into `(w, h)` array of `RGB`
"""
function channels_to_rgb(im)
    im_rgb_unwinded = [RGB(im[i, j, :]...) for j in range(1, size(im, 2)) for i in range(1, size(im, 1))]
    im_rgb = reshape(im_rgb_unwinded, size(im, 1), size(im, 2))'
end