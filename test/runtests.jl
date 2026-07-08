using Test
t1 = @elapsed using cDFT
using Clapeyron
using GCIdentifier
using ChemicalIdentifiers
@info "Loading cDFT took $(round(t1,digits = 2)) seconds"

@testset "cDFT" begin
    include("test_models.jl")
    include("test_methods.jl")
    include("test_coordinate_systems.jl")
    include("test_morphology.jl")
    include("test_scft.jl")
end