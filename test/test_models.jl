using Test, cDFT
using cDFT.Clapeyron

@testset "Models" begin
    p = 1e5
    T = 298.15
    x = [0.5,0.5]
    x3 = [0.333, 0.333,0.333]

    @testset "PCSAFT" begin
        model = PCSAFT(["water","ethanol"])

        μ1 = Clapeyron.chemical_potential_res(model,p,T,x)/T/Clapeyron.Rgas()
        vl = volume(model, p, T, x)
        ρ = x/vl
        
        L = cDFT.length_scale(model)
        
        structure = Uniform1DCart((p, T), ρ,[-10L,10L], (3,))
        system = DFTSystem(model, structure)
        ρ = cDFT.initialize_profiles(system)
        μ2 = cDFT.δFδρ_res(system, ρ)

        @test μ1[1] ≈ μ2[1] rtol = 1e-6

        # structure = Uniform1DSphr((p, T, x),[2L,20L], 3)
        # system = DFTSystem(model, structure)
        # μ3 = cDFT.δFδρ_res(system)

        # @test μ1[1] ≈ μ3[1] rtol = 1e-6
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

    @testset "SAFTVRMie" begin
        model = SAFTVRMie(["water","methanol"])

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
end