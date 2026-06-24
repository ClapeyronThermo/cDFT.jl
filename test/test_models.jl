using Test, cDFT
using cDFT.Clapeyron

@testset "Models" begin
    p = 1e5
    T = 298.15
    x = [0.5,0.5]
    x1 = [1.]
    x3 = [0.333, 0.333,0.333]

    @testset "PCSAFT" begin
        model = PCSAFT(["hexane","acetone"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = cDFT.length_scale(model)
        
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6

    end

    @testset "PCPSAFT" begin
        model = PCPSAFT(["acetone","hexane","DMSO"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x3)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x3)
        ρ = x3/vl
        
        L = cDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "HeterogcPCPSAFT" begin
        # Use non-associating, non-polar alkanes: GCIdentifier supports CH3/CH2 for gcPCSAFT.
        # acetone contains >C=O which GCIdentifier cannot map → bond topology error.
        model = HeterogcPCPSAFT(["hexane","heptane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = cDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "SAFTVRMie" begin
        model = SAFTVRMie(["methane","butane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = cDFT.length_scale(model)
        structure = Uniform1DCart((p, T),ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "SAFTgammaMie" begin
        # Use non-associating species: methane (1 bead) + butane (4 beads).
        # Association contributions are deferred; checking HS+chain+disp only.
        model = SAFTgammaMie(["methane","butane"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl

        L = cDFT.length_scale(model)
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end

    @testset "COFFEE" begin
        model = COFFEE(["a1"]; userlocations = (;
            Mw = [1.],
            segment = [1.],
            sigma = [3.;;],
            epsilon = [300.;;],
            lambda_r = [12;;],
            lambda_a = [6;;],
            shift = [3*0.15],
            dipole = [1.0*1.0575091914494172],))

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x1)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x1)
        ρ = x1/vl
        
        L = cDFT.length_scale(model)
        structure = Uniform1DCart((p, T),ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6
    end
end