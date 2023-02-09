module KeypointsDetection

include("facial_keypoints/Utils.jl")
include("facial_keypoints/NetDefinitions.jl")
include("facial_keypoints/TrainUtils.jl")
include("PredictUtils.jl")
include("facial_keypoints/PredictUtilsBasic.jl")

using Requires

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("facial_keypoints/PlotUtils.jl")
    end
    @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("facial_keypoints/Augmentation.jl")
    @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("facial_keypoints/LoadDataset.jl")
    @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("facial_keypoints/ValidateUtils.jl")
    @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("facial_keypoints/DataFramePredictUtils.jl")
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" begin
        @require VideoIO="d6d074c3-1acf-5d4c-9a43-ef38773959a2" begin
            @require ObjectDetector="3dfc1049-5314-49cf-8447-288dfd02f9fb" include("WebCamUtils.jl")
        end
    end
    @require ObjectDetector="3dfc1049-5314-49cf-8447-288dfd02f9fb" include("YOLOUtils.jl")
end

end
