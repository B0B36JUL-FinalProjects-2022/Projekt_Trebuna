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
