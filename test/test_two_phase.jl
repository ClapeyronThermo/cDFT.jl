using Test, cDFT
using cDFT.Clapeyron

@testset "TwoPhase" begin
    model = PCSAFT(["methane"])
    T = 150.0
    (p, vl, vv) = Clapeyron.saturation_pressure(model, T)
    ρl, ρv = [1/vl], [1/vv]
    L = cDFT.length_scale(model)

    # Only TwoPhase1DCart is exercised anywhere else in the repo today (via
    # surface_tension/interfacial_tension, both of which only ever regression-check one
    # derived scalar — never the profile itself). The other 5 structures are otherwise
    # completely unused outside their own definitions.
    function check_two_phase(system)
        ρ = cDFT.initialize_profiles(system)
        @test all(ρ .> 0)
        # Genuinely spans between the two bulk phases (not collapsed to a single value).
        @test maximum(ρ) - minimum(ρ) > 0.5 * abs(ρl[1] - ρv[1])
        return ρ
    end

    @testset "TwoPhase1DCart" begin
        structure = TwoPhase1DCart((p, T), ρl, ρv, [-10L, 10L], 51)
        check_two_phase(DFTSystem(model, structure))
    end

    @testset "TwoPhase2DLamCart" begin
        structure = cDFT.TwoPhase2DLamCart((p, T), ρl, ρv, [-10L 10L; -1L 1L], (9, 5))
        check_two_phase(DFTSystem(model, structure))
    end

    @testset "TwoPhase3DLamCart" begin
        structure = cDFT.TwoPhase3DLamCart((p, T), ρl, ρv, [-10L 10L; -1L 1L; -1L 1L], (9, 5, 5))
        check_two_phase(DFTSystem(model, structure))
    end

    @testset "TwoPhase2DHexCart" begin
        structure = TwoPhase2DHexCart((p, T), ρl, ρv, [-10L, 10L], (9,))
        check_two_phase(DFTSystem(model, structure))
    end

    @testset "TwoPhase3DHexCart" begin
        structure = cDFT.TwoPhase3DHexCart((p, T), ρl, ρv, [-10L, 10L], (9,))
        check_two_phase(DFTSystem(model, structure))
    end

    @testset "TwoPhase3DSphrCart" begin
        structure = cDFT.TwoPhase3DSphrCart((p, T), ρl, ρv, [-10L, 10L], (9,))
        check_two_phase(DFTSystem(model, structure))
    end
end
