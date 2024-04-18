@testset "Methods" begin
    @testset "Surface Tension" begin
        model = PCSAFT(["ethanol","hexane"])

        T = 298.15
        x = [0.5,0.5]

        γ1 = surface_tension(model,T,x)

        @test γ1 ≈ 0.019843696660439683 rtol = 1e-6

        p,vl,vv,y = Clapeyron.bubble_pressure(model,T,x)

        L = cDFT.length_scale(model)
        structure = SurfaceTension1DCart((p, T, x),[-10L,10L], 101)
        system = DFTSystem(model, structure)
        converge!(system)

        γ2 = cDFT.surface_tension(system)

        @test γ1 ≈ γ2 rtol = 1e-6
    end

    @testset "Interfacial Tension" begin
        model = PCSAFT(["water","hexane"])

        p = 1e5
        T = 298.15
        n = [0.5,0.5]

        γ1 = interfacial_tension(model,p,T,n)

        @test γ1 ≈ 0.030742209755905244 rtol = 1e-6

        L = cDFT.length_scale(model)

        structure = InterfacialTension1DCart((p, T, n),[-10L,10L], 201)

        system = DFTSystem(model, structure)

        converge!(system)

        γ2 = interfacial_tension(system)

        @test γ1 ≈ γ2 rtol = 1e-6
    end
end