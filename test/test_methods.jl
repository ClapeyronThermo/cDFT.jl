@testset "Methods" begin
    @testset "Surface Tension" begin
        model = PCSAFT(["ethanol","hexane"])

        T = 298.15
        x = [0.5,0.5]

        γ1 = surface_tension(model,T,x)

        @test γ1 ≈ 0.019843355568868397 rtol = 1e-4
    end

    @testset "Interfacial Tension" begin
        model = PCSAFT(["water","hexane"])

        p = 1e5
        T = 298.15
        n = [0.5,0.5]

        (x,_,_) = tp_flash(model, 1e5, 298.15, [0.5,0.5], RRTPFlash(equilibrium=:lle))

        γ1 = interfacial_tension(model, p, T, x[1,:], x[2,:])

        @test γ1 ≈ 0.03073749588489727 rtol = 1e-4
    end

    @testset "Adsorption" begin
        model = PCSAFT(["carbon dioxide","methane"])
        surface = Steele(["graphite"],40e-10)

        p = 1e6
        T = 298.15
        n = [0.5,0.5]

        ad = adsorption(model, surface, p, T, n)

        @test ad[1] ≈ 598.4454929600739 rtol = 1e-4
    end
end