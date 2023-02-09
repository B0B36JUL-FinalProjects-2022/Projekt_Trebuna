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
