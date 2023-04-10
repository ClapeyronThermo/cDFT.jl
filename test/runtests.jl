using Test, cDFT
using cDFT.Clapeyron

@testset "cDFT" begin
    model = PCSAFT(["water"])
    surface_tension(model,372.9) ≈ 0.07633680532414205 rtol = 1e-6
end