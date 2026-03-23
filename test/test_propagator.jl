using FFTW
using KernelAbstractions

@testset "DiscreteGaussianChainPropagator" begin

    # Helper: minimal mock system for propagator tests
    struct MockSystem{S,P}
        structure::S
        propagator::P
    end
    cDFT.dimension(s::MockSystem) = cDFT.dimension(s.structure)

    backend = CPU()

    @testset "Kernel values" begin
        L = 10.0
        ngrid = 64
        structure = cDFT.Uniform1DCart((1e5, 298.15), [1.0], [0.0, L], ngrid)

        b_val = 1.0
        prop = cDFT.DiscreteGaussianChainPropagator(
            [b_val], [3], [[1, 1, 1]], structure, backend
        )

        kernel = prop.kernel_map[(1, 1)]

        # Verify kernel at ν=0 is 1.0
        @test real(kernel[1]) ≈ 1.0 atol = 1e-14

        # Verify kernel at known frequency
        ν1 = 1.0 / L
        expected = exp(-2π^2 * b_val^2 * ν1^2 / 3)
        @test real(kernel[2]) ≈ expected rtol = 1e-12
    end

    @testset "Junction kernel values" begin
        L = 10.0
        ngrid = 64
        structure = cDFT.Uniform1DCart((1e5, 298.15), [1.0, 1.0], [0.0, L], ngrid)

        b_A = 1.0
        b_B = 2.0
        # A₂B₂ diblock: segments [1,1,2,2]
        prop = cDFT.DiscreteGaussianChainPropagator(
            [b_A, b_B], [4], [[1, 1, 2, 2]], structure, backend
        )

        # Should have 3 kernels: (1,1), (1,2), (2,2)
        @test haskey(prop.kernel_map, (1, 1))
        @test haskey(prop.kernel_map, (1, 2))
        @test haskey(prop.kernel_map, (2, 2))

        # Verify AA kernel at ν=0
        @test real(prop.kernel_map[(1, 1)][1]) ≈ 1.0 atol = 1e-14
        # Verify BB kernel at ν=0
        @test real(prop.kernel_map[(2, 2)][1]) ≈ 1.0 atol = 1e-14
        # Verify junction kernel at ν=0
        @test real(prop.kernel_map[(1, 2)][1]) ≈ 1.0 atol = 1e-14

        # Verify junction kernel uses b_AB = √((b_A² + b_B²)/2)
        b_AB = sqrt((b_A^2 + b_B^2) / 2)
        ν1 = 1.0 / L
        expected_AA = exp(-2π^2 * b_A^2 * ν1^2 / 3)
        expected_BB = exp(-2π^2 * b_B^2 * ν1^2 / 3)
        expected_AB = exp(-2π^2 * b_AB^2 * ν1^2 / 3)

        @test real(prop.kernel_map[(1, 1)][2]) ≈ expected_AA rtol = 1e-12
        @test real(prop.kernel_map[(2, 2)][2]) ≈ expected_BB rtol = 1e-12
        @test real(prop.kernel_map[(1, 2)][2]) ≈ expected_AB rtol = 1e-12
    end

    @testset "Uniform field - single homopolymer" begin
        L = 10.0
        ngrid = 64
        structure = cDFT.Uniform1DCart((1e5, 298.15), [1.0], [0.0, L], ngrid)

        N_seg = 5
        b_val = 1.0
        prop = cDFT.DiscreteGaussianChainPropagator(
            [b_val], [N_seg], [[1, 1, 1, 1, 1]], structure, backend
        )

        system = MockSystem(structure, prop)

        nspecies = 1
        ρ = ones(ngrid, nspecies)
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, prop, ρ, backend)

        w0 = 0.5
        δfδρ_res = fill(w0, ngrid, nspecies)

        cDFT.propagate!(system, prop, ρ, δfδρ_res, q_fwd, q_bwd, buf, P, iP)

        # With uniform field, convolution preserves uniform profiles (kernel(0)=1)
        # q_fwd(i) = exp(-i*w0): each step multiplies by exp(-w0) after convolving
        for i in 1:N_seg
            @test all(x -> isapprox(x, exp(-i * w0); atol=1e-12), selectdim(q_fwd[1], 2, i))
            @test all(x -> isapprox(x, exp(-i * w0); atol=1e-12), selectdim(q_bwd[1], 2, i))
        end

        # Product q_fwd(j)*q_bwd(N+1-j) = exp(-j*w0)*exp(-(N+1-j)*w0) = exp(-(N+1)*w0)
        # Sum over N segments = N * exp(-(N+1)*w0)
        # δfδρ_res = w0 - log(N * exp(-(N+1)*w0)) = w0 - log(N) + (N+1)*w0 = (N+2)*w0 - log(N)
        expected_field = (N_seg + 2) * w0 - log(N_seg)
        @test all(x -> isapprox(x, expected_field; atol=1e-10), δfδρ_res)
    end

    @testset "Multi-species diblock copolymer" begin
        L = 10.0
        ngrid = 64
        structure = cDFT.Uniform1DCart((1e5, 298.15), [1.0, 1.0], [0.0, L], ngrid)

        # A₃B₂ diblock: segments [1,1,1,2,2]
        N_seg = 5
        b_val = 1.0
        seg_spec = [1, 1, 1, 2, 2]
        prop = cDFT.DiscreteGaussianChainPropagator(
            [b_val, b_val], [N_seg], [seg_spec], structure, backend
        )

        system = MockSystem(structure, prop)

        nspecies = 2
        ρ = ones(ngrid, nspecies)
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, prop, ρ, backend)

        w_A = 0.3
        w_B = 0.7
        δfδρ_res = hcat(fill(w_A, ngrid), fill(w_B, ngrid))

        cDFT.propagate!(system, prop, ρ, δfδρ_res, q_fwd, q_bwd, buf, P, iP)

        exp_wA = exp(-w_A)
        exp_wB = exp(-w_B)

        # Forward: q(1)=e^{-wA}, q(2)=e^{-2wA}, q(3)=e^{-3wA}, q(4)=e^{-wB-3wA}, q(5)=e^{-2wB-3wA}
        @test all(x -> isapprox(x, exp_wA; atol=1e-12), selectdim(q_fwd[1], 2, 1))
        @test all(x -> isapprox(x, exp_wA^2; atol=1e-12), selectdim(q_fwd[1], 2, 2))
        @test all(x -> isapprox(x, exp_wA^3; atol=1e-12), selectdim(q_fwd[1], 2, 3))
        @test all(x -> isapprox(x, exp_wB * exp_wA^3; atol=1e-12), selectdim(q_fwd[1], 2, 4))
        @test all(x -> isapprox(x, exp_wB^2 * exp_wA^3; atol=1e-12), selectdim(q_fwd[1], 2, 5))

        # Backward: q†(1)=e^{-wB}, q†(2)=e^{-2wB}, q†(3)=e^{-wA-2wB}, q†(4)=e^{-2wA-2wB}, q†(5)=e^{-3wA-2wB}
        @test all(x -> isapprox(x, exp_wB; atol=1e-12), selectdim(q_bwd[1], 2, 1))
        @test all(x -> isapprox(x, exp_wB^2; atol=1e-12), selectdim(q_bwd[1], 2, 2))
        @test all(x -> isapprox(x, exp_wA * exp_wB^2; atol=1e-12), selectdim(q_bwd[1], 2, 3))
        @test all(x -> isapprox(x, exp_wA^2 * exp_wB^2; atol=1e-12), selectdim(q_bwd[1], 2, 4))
        @test all(x -> isapprox(x, exp_wA^3 * exp_wB^2; atol=1e-12), selectdim(q_bwd[1], 2, 5))

        # Products q_fwd(j)*q_bwd(N+1-j):
        # j=1: e^{-wA} * e^{-3wA-2wB} = e^{-4wA-2wB}
        # j=2: e^{-2wA} * e^{-2wA-2wB} = e^{-4wA-2wB}
        # j=3: e^{-3wA} * e^{-wA-2wB} = e^{-4wA-2wB}
        # j=4: e^{-wB-3wA} * e^{-2wB} = e^{-3wA-3wB}
        # j=5: e^{-2wB-3wA} * e^{-wB} = e^{-3wA-3wB}
        sum_A = 3 * exp(-4w_A - 2w_B)
        sum_B = 2 * exp(-3w_A - 3w_B)

        expected_A = w_A - log(sum_A)
        expected_B = w_B - log(sum_B)

        @test all(x -> isapprox(x, expected_A; atol=1e-10), selectdim(δfδρ_res, 2, 1))
        @test all(x -> isapprox(x, expected_B; atol=1e-10), selectdim(δfδρ_res, 2, 2))
    end

    @testset "3D uniform field" begin
        ngrid = 16
        L = 5.0
        bounds_3d = [0.0 L; 0.0 L; 0.0 L]
        structure = cDFT.Uniform3DCart((1e5, 298.15), [1.0], bounds_3d, ngrid)

        N_seg = 3
        b_val = 0.5
        prop = cDFT.DiscreteGaussianChainPropagator(
            [b_val], [N_seg], [[1, 1, 1]], structure, backend
        )

        system = MockSystem(structure, prop)

        nspecies = 1
        ρ = ones(ngrid, ngrid, ngrid, nspecies)
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, prop, ρ, backend)

        w0 = 0.4
        δfδρ_res = fill(w0, ngrid, ngrid, ngrid, nspecies)

        cDFT.propagate!(system, prop, ρ, δfδρ_res, q_fwd, q_bwd, buf, P, iP)

        # Same accumulation as 1D: q_fwd(i) = exp(-i*w0)
        for i in 1:N_seg
            @test all(x -> isapprox(x, exp(-i * w0); atol=1e-10), selectdim(q_fwd[1], 4, i))
        end

        expected_field = (N_seg + 2) * w0 - log(N_seg)
        @test all(x -> isapprox(x, expected_field; atol=1e-10), δfδρ_res)
    end

    @testset "Multiple chains" begin
        L = 10.0
        ngrid = 64
        structure = cDFT.Uniform1DCart((1e5, 298.15), [1.0, 1.0], [0.0, L], ngrid)

        # Chain 1: homopolymer A with 3 segments, b=1.0
        # Chain 2: homopolymer B with 4 segments, b=0.5
        prop = cDFT.DiscreteGaussianChainPropagator(
            [1.0, 0.5],
            [3, 4],
            [[1, 1, 1], [2, 2, 2, 2]],
            structure,
            backend
        )

        system = MockSystem(structure, prop)

        nspecies = 2
        ρ = ones(ngrid, nspecies)
        q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, prop, ρ, backend)

        w1 = 0.3
        w2 = 0.6
        δfδρ_res = hcat(fill(w1, ngrid), fill(w2, ngrid))

        cDFT.propagate!(system, prop, ρ, δfδρ_res, q_fwd, q_bwd, buf, P, iP)

        # Chain 1 (species 1, N=3): products = 3*exp(-(3+1)*w1)
        # Chain 2 (species 2, N=4): products = 4*exp(-(4+1)*w2)
        # δfδρ_res[1] = w1 - log(3*exp(-4*w1))
        # δfδρ_res[2] = w2 - log(4*exp(-5*w2))
        expected_1 = w1 - log(3 * exp(-4w1))
        expected_2 = w2 - log(4 * exp(-5w2))

        @test all(x -> isapprox(x, expected_1; atol=1e-10), selectdim(δfδρ_res, 2, 1))
        @test all(x -> isapprox(x, expected_2; atol=1e-10), selectdim(δfδρ_res, 2, 2))
    end
end
