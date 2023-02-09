using KeypointsDetection
using Test
using DataFrames

@testset "DatasetUtils" begin
    dataframe = DataFrame(
        A = [1, 2, 3, missing],
        B = [missing, 2, 3, 4],
        C = [missing, missing, missing, missing]
    )
    colNames1, nrows1 = ["A", "B"], 2
    colNames2, nrows2 = ["C"], 0
    colNames3, nrows3 = ["A"], 3
    @testset "Correct rows removed" for (colNames, num_rows) in zip(
        [colNames1, colNames2, colNames3],
        [nrows1, nrows2, nrows3]
    )
        newDataframe = KeypointsDetection.droprows_wo_needed_columns(dataframe, colNames)
        @test nrow(newDataframe) == num_rows
    end
end

@testset "AugmentationUtils" begin
    @testset "Horizontal flip" begin
        dataframe = DataFrame(
            Image=[
                [1 2; 3 4],
                [1 2; 3 4],
                [1 2; 3 4]
            ],
            A=[10, 20, 30],
            B=[40, 50, 60],
            C=[60, 70, 80]
        )

        newDataframe = horizontal_flip(
            dataframe,
            indices=[2, 4],
            notIndices=[3],
            new_indices=[4, 2],
            new_notIndices=[3]
        )
        @test all(newDataframe[i, :Image] == [2 1; 4 3] for i in 1:3)
        @test newDataframe[:, :A] == [96 - 60, 96-70, 96-80]
        @test newDataframe[:, :C] == [96 - 10, 96 - 20, 96 - 30]
        @test newDataframe[:, :B] == dataframe[:, :B]
    end
end

@testset "TransformPredictions" begin
    preds = [
        (65.52859, 40.049026),
        (29.3091, 36.508976),
        (44.74062, 61.655544),
        (43.800518, 76.93857),
    ]

    new_preds = [
        (358.4629751046499, 116.17168315251669),
        (293.19243917862576, 109.79221713542938),
        (321.00132501125336, 155.108428756396),
        (319.30718354384106, 182.6497112909953),
    ]

    old_x, old_y, r, p11, p21 = 262, 44, 0.5549132947976878, 12, 0
    res = [(p[1] - np[1]) + (p[2] - np[2]) for (p, np) in zip(preds_to_full(preds, old_x, old_y, r, p11, p21), new_preds)]
    res = sum(res) / size(res, 1)
    @test res ≈ 0 atol=1e-4
end
