using Test
t1 = @elapsed using cDFT
using cDFT.Clapeyron
@info "Loading cDFT took $(round(t1,digits = 2)) seconds"

@testset "cDFT" begin
    include("test_models.jl")
    include("test_methods.jl")
end