export sort_by_error

"""
Calculate mean squared error of preds and gold.

Args:
- `preds`: the predictions
- `gold`: the gold labels
"""
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

"""
Calculate mean squared error of each row pair(preds[row, :], gold[row, :]), then
create a dataframe with additional column `:Error` with mean squared error quantity.
Sort the resulting dataframe according to `:Error` and store the original index of
each row in the column `:TrueIndex`.
"""
function sort_by_error(predictedDataframe::DataFrame, goldDataframe::DataFrame)
    df = deepcopy(predictedDataframe)
    df[!, :Error] = [calculateError(p, g) for (p, g) in zip(eachrow(predictedDataframe), eachrow(goldDataframe))]
    df[!, :TrueIndex] = collect(range(1, DataFrames.nrow(df)))
    sort!(df, [:Error])
    df
end
