function calculateError(preds, gold)
    total, e = 0, 0
    for (p, g) in zip(preds, gold)
        if g isa Matrix
            break
        end
        if !ismissing(g) && !ismissing(p)
            total += 1
            e += (p - g)^2
        end
    end
    return e / total
end

function sort_by_error(predictedDataframe::DataFrame, goldDataframe::DataFrame)
    df = deepcopy(predictedDataframe)
    df[!, :Error] = [calculateError(p, g) for (p, g) in zip(eachrow(predictedDataframe), eachrow(goldDataframe))]
    df[!, :TrueIndex] = collect(range(1, DataFrames.nrow(df)))
    sort!(df, [:Error])
    df
end
