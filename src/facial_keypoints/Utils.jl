"""
Measures the time needed to run the `method` argument, returns the time
"""
function measure_time(method)
    s = time()
    method()
    e = time()
    e - s
end