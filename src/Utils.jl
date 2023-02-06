
function measure_time(method)
    s = time()
    method()
    e = time()
    e - s
end