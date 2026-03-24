using FFTW
using KernelAbstractions
using LinearAlgebra

@testset "SCFT" begin

    backend = CPU()

    @testset "FloryHuggins constructor" begin
        chi = [0.0 10.0; 10.0 0.0]
        fh = cDFT.FloryHuggins(chi, 1.0, 20.0)
        @test fh.chi == chi
        @test fh.rho0 == 1.0
        @test fh.kappa == 20.0
    end

    @testset "Field computation - FloryHuggins" begin
        nspecies = 2
        chi = [0.0 10.0; 10.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        L = 10.0
        ngrid = 64

        chain_A = cDFT.SCFTChain(N=5, b=1.0, segment_species=[1,1,1,2,2],
                                  ensemble=:canonical, n_chains=10.0)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain_A],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        # Uniform density test
        ρ_A = 0.6
        ρ_B = 0.4
        ρ = hcat(fill(ρ_A, ngrid), fill(ρ_B, ngrid))
        w = similar(ρ)

        cDFT.compute_fields!(system, ρ, w)

        # Expected: w_A = χ_AB/ρ₀ * ρ_B + ζ/ρ₀ * (ρ₊/ρ₀ - 1)
        ρ_total = ρ_A + ρ_B
        comp = (kappa / rho0) * (ρ_total / rho0 - 1.0)
        expected_wA = chi[1,2] / rho0 * ρ_B + comp
        expected_wB = chi[2,1] / rho0 * ρ_A + comp

        @test all(x -> isapprox(x, expected_wA; atol=1e-12), w[:, 1])
        @test all(x -> isapprox(x, expected_wB; atol=1e-12), w[:, 2])
    end

    @testset "Bulk fields" begin
        chi = [0.0 10.0 5.0; 10.0 0.0 3.0; 5.0 3.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        bulk = [0.4, 0.3, 0.3]
        w_bulk = cDFT.compute_bulk_fields(fh, bulk)

        ρ_total = sum(bulk)
        comp = (kappa / rho0) * (ρ_total / rho0 - 1.0)

        @test w_bulk[1] ≈ chi[1,2]/rho0 * bulk[2] + chi[1,3]/rho0 * bulk[3] + comp
        @test w_bulk[2] ≈ chi[2,1]/rho0 * bulk[1] + chi[2,3]/rho0 * bulk[3] + comp
        @test w_bulk[3] ≈ chi[3,1]/rho0 * bulk[1] + chi[3,2]/rho0 * bulk[2] + comp
    end

    @testset "Uniform self-consistency" begin
        # A uniform system at bulk densities should be a fixed point:
        # fields(ρ_bulk) → propagators → Q → densities should return ρ_bulk
        nspecies = 2
        chi = [0.0 1.0; 1.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        L = 10.0
        ngrid = 65  # odd for Simpson rule

        N_seg = 10
        n_chains = 1.0
        seg_spec = vcat(fill(1, 5), fill(2, 5))
        chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                                ensemble=:canonical, n_chains=n_chains)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        bulk = cDFT.compute_bulk_densities(system)
        ρ_A_bulk = bulk[1]
        ρ_B_bulk = bulk[2]

        ρ = hcat(fill(ρ_A_bulk, ngrid), fill(ρ_B_bulk, ngrid))
        w = similar(ρ)
        ρ_new = similar(ρ)

        cDFT.compute_fields!(system, ρ, w)

        propagator = system.propagator
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, propagator, ρ, backend)
        w_bulk = cDFT.compute_bulk_fields(fh, bulk)
        cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)

        dz = cDFT.structure_dz(system.structure)
        Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz)

        cDFT.compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ_new)

        # For a uniform system, the new densities should match the input
        @test all(x -> isapprox(x, ρ_A_bulk; rtol=1e-6), ρ_new[:, 1])
        @test all(x -> isapprox(x, ρ_B_bulk; rtol=1e-6), ρ_new[:, 2])
    end

    @testset "Free energy - uniform system" begin
        nspecies = 2
        chi = [0.0 1.0; 1.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        L = 10.0
        ngrid = 65

        N_seg = 10
        n_chains = 1.0
        seg_spec = vcat(fill(1, 5), fill(2, 5))
        chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                                ensemble=:canonical, n_chains=n_chains)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        bulk = cDFT.compute_bulk_densities(system)
        ρ_A_bulk = bulk[1]
        ρ_B_bulk = bulk[2]

        ρ = hcat(fill(ρ_A_bulk, ngrid), fill(ρ_B_bulk, ngrid))
        w = similar(ρ)
        cDFT.compute_fields!(system, ρ, w)

        propagator = system.propagator
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, propagator, ρ, backend)
        w_bulk = cDFT.compute_bulk_fields(fh, bulk)
        cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
        dz = cDFT.structure_dz(system.structure)
        Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz)

        H = cDFT.free_energy(system, ρ, w, Q_chains, Q_solvents)

        # The free energy should be a finite number for a well-defined system
        @test isfinite(H)

        # For uniform system, U_int = χ_AB * ρ_A * ρ_B * V_eff / ρ₀
        V_eff = cDFT.effective_volume(system, dz)
        U_int_expected = chi[1,2] * ρ_A_bulk * ρ_B_bulk * V_eff / rho0
        ρ_total = ρ_A_bulk + ρ_B_bulk
        U_comp_expected = (kappa / (2.0 * rho0)) * (ρ_total / rho0 - 1.0)^2 * V_eff

        @test U_int_expected > 0
        @test isfinite(U_comp_expected)
    end

    @testset "Initialize profiles" begin
        nspecies = 2
        chi = [0.0 10.0; 10.0 0.0]
        fh = cDFT.FloryHuggins(chi, 1.0, 20.0)

        L = 10.0
        ngrid = 64
        n_chains = 10.0

        chain = cDFT.SCFTChain(N=10, b=1.0, segment_species=vcat(fill(1,5), fill(2,5)),
                                ensemble=:canonical, n_chains=n_chains)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        # Uniform initialization
        ρ = cDFT.initialize_profiles(system; mode=:uniform)
        @test size(ρ) == (ngrid, nspecies)
        @test all(ρ[:, 1] .== ρ[1, 1])
        @test all(ρ[:, 2] .== ρ[1, 2])

        # Perturbed initialization
        ρ_pert = cDFT.initialize_profiles(system; mode=:perturbed)
        @test size(ρ_pert) == (ngrid, nspecies)
        @test !all(ρ_pert[:, 1] .== ρ_pert[1, 1])
    end

    @testset "Grand canonical solvent" begin
        nspecies = 3  # A, B, S
        chi = [0.0 10.0 5.0; 10.0 0.0 3.0; 5.0 3.0 0.0]
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        L = 10.0
        ngrid = 65

        N_seg = 10
        seg_spec = vcat(fill(1, 5), fill(2, 5))
        chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                                ensemble=:canonical, n_chains=10.0)

        solvent = cDFT.SCFTSolvent(species_index=3, ensemble=:grand_canonical,
                                    bulk_density=0.3)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            solvents=[solvent],
            nspecies=nspecies,
            species_names=[:A, :B, :S],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        bulk = cDFT.compute_bulk_densities(system)
        w_bulk = cDFT.compute_bulk_fields(fh, bulk)

        ρ = zeros(ngrid, nspecies)
        for α in 1:nspecies
            ρ[:, α] .= bulk[α]
        end
        w = similar(ρ)
        cDFT.compute_fields!(system, ρ, w)

        ρ_new = similar(ρ)
        propagator = system.propagator
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, propagator, ρ, backend)
        cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
        dz = cDFT.structure_dz(system.structure)
        Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz)
        cDFT.compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ_new)

        # Solvent density should match bulk
        @test all(x -> isapprox(x, 0.3; rtol=1e-6), ρ_new[:, 3])
    end

    @testset "3D SCFTSystem construction" begin
        nspecies = 2
        chi = [0.0 10.0; 10.0 0.0]
        fh = cDFT.FloryHuggins(chi, 1.0, 20.0)

        L = 5.0
        ngrid = 16
        bounds_3d = [0.0 L; 0.0 L; 0.0 L]

        chain = cDFT.SCFTChain(N=4, b=1.0, segment_species=[1,1,2,2],
                                ensemble=:canonical, n_chains=10.0)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=bounds_3d,
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        @test cDFT.dimension(system) == 3
        @test system.structure.ngrid == (ngrid, ngrid, ngrid)

        ρ = cDFT.initialize_profiles(system; mode=:uniform)
        @test size(ρ) == (ngrid, ngrid, ngrid, nspecies)
    end

    @testset "Convergence - perturbed to uniform" begin
        # A symmetric AB diblock with weak χ should converge back to uniform
        # from a small perturbation (below the spinodal)
        nspecies = 2
        chi = [0.0 0.5; 0.5 0.0]  # χN = 5, well below ODT (~10.5)
        rho0 = 1.0
        kappa = 20.0
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        L = 10.0
        ngrid = 65

        N_seg = 10
        n_chains = 1.0
        seg_spec = vcat(fill(1, 5), fill(2, 5))
        chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                                ensemble=:canonical, n_chains=n_chains)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        bulk = cDFT.compute_bulk_densities(system)

        # Perturbed initial condition
        ρ = cDFT.initialize_profiles(system; mode=:perturbed, perturbation=0.05)

        # Verify it's actually perturbed
        @test maximum(abs.(ρ[:, 1] .- bulk[1])) > 0.001

        # Run field-based convergence
        result = cDFT.converge_fields!(system, ρ)

        # Should converge back to uniform bulk densities
        @test all(x -> isapprox(x, bulk[1]; rtol=1e-3), ρ[:, 1])
        @test all(x -> isapprox(x, bulk[2]; rtol=1e-3), ρ[:, 2])
    end

    @testset "Convergence - lamellar microphase separation" begin
        # Symmetric AB diblock above the ODT (χN=30 >> 10.5) should form lamellae
        nspecies = 2
        N_seg = 20
        chi_val = 1.5  # χN = 30
        rho0 = 1.0
        kappa = 20.0
        chi = [0.0 chi_val; chi_val 0.0]
        fh = cDFT.FloryHuggins(chi, rho0, kappa)

        # Box ≈ one lamellar period (Rg ≈ b√(N/6) ≈ 1.83, d ≈ 3.8 Rg ≈ 7)
        L = 7.0
        ngrid = 65
        n_chains = L / N_seg  # total segment density ≈ ρ₀

        seg_spec = vcat(fill(1, N_seg÷2), fill(2, N_seg÷2))
        chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                                ensemble=:canonical, n_chains=n_chains)

        system = cDFT.SCFTSystem(
            interaction=fh,
            chains=[chain],
            nspecies=nspecies,
            species_names=[:A, :B],
            bounds=[0.0, L],
            ngrid=ngrid,
            options=cDFT.DFTOptions()
        )

        bulk = cDFT.compute_bulk_densities(system)

        # Seed with sinusoidal perturbation to break symmetry toward lamellae
        ρ = cDFT.initialize_profiles(system; mode=:uniform)
        z = range(0, L, length=ngrid)
        amp = 0.1 * bulk[1]
        for i in 1:ngrid
            ρ[i, 1] += amp * cos(2π * z[i] / L)
            ρ[i, 2] -= amp * cos(2π * z[i] / L)
        end
        clamp!(ρ, 1e-10, Inf)

        result = cDFT.converge_fields!(system, ρ)

        # Density profiles should be strongly non-uniform (microphase separated)
        amp_A = maximum(ρ[:, 1]) - minimum(ρ[:, 1])
        amp_B = maximum(ρ[:, 2]) - minimum(ρ[:, 2])
        @test amp_A > 0.5  # large compositional contrast
        @test amp_B > 0.5

        # A and B should be out of phase (complementary domains)
        @test argmax(ρ[:, 1]) != argmax(ρ[:, 2])

        # Incompressibility: total density ≈ ρ₀ everywhere
        ρ_total = ρ[:, 1] .+ ρ[:, 2]
        @test all(x -> isapprox(x, rho0; rtol=0.05), ρ_total)
    end
end
