using GeometryBasics
using .DataFrames

function predict_to_dataframe(net::NetHolder, dataframe::DataFrame; needed_columns=names(dataframe), withgpu=false)
    X = create_predict_dataset(dataframe)
    if !("Image" in needed_columns)
        push!(needed_columns, "Image")
    end 
    predict_to_dataframe(net, X, dataframe, needed_columns, withgpu)
end

function predict_to_dataframe(net::NetHolder, X::AbstractArray, dataframe::DataFrame, needed_columns, withgpu)
    preds = predict(net, X; withgpu)
    df = DataFrame()
    i = 1
    for name in names(dataframe)
        if name == "Image"
            df[!, name] = dataframe[!, name]
            break
        end
        if name in needed_columns
            df[!, name] = preds[i, :]
            i += 1
        else
            df[!, name] = [missing for _ in range(1, size(preds)[2])]
        end
    end
    df
end