module SciMLBasecDFTExt
    import cDFT
    import SciMLBase
    import KernelAbstractions as KA
    import FFTW


    ################### DYNAMIC DENSITY FUNCTIONAL THEORY ###################

    function SciMLBase.ODEProblem(system::cDFT.AbstractcDFTSystem, ρ, tspan, kwargs...)
        k        = cDFT.structure_ω(system.structure, system.options.device) .* cDFT.length_scale(system.model)

        ngrid    = system.structure.ngrid

        nd       = length(ngrid)

        map_grad =  2π .* k .* im
        map_lapl = -(2π)^2 .* sum(k.^2, dims=nd+1)

        tmp      = KA.allocate(system.options.device, Float64, ngrid...)
        buf      = KA.allocate(system.options.device, ComplexF64, ngrid...)
        buf_real = similar(tmp, Float64)
        # ρ        = similar(ρ0, Float64)
        P        = FFTW.plan_fft!(buf)
        iP       = FFTW.plan_ifft!(buf)
        
        μ, cache_model, cache_external_field, cache_propagator = cDFT.preallocate_newautodiff(system, ρ)

        function ddft_rhs_log!(dη, η, params, t)
            @. ρ = exp(clamp(η, -100, 50))
            # println(t, " ", minimum(ρ), " ", maximum(ρ))
            # println(t, " ", minimum(μ), " ", maximum(μ))

            cDFT.δFδρ_res_newautodiff!(system, ρ, μ, cache_model...)
            # println(t, " ", minimum(μ), " ", maximum(μ))
            cDFT.evaluate_external_field!(system, ρ, μ, cache_external_field)
            cDFT.propagate!(system, ρ, μ, cache_propagator)

            for α in axes(η, nd + 1)
                η_α  = selectdim(η,  nd + 1, α)
                μ_α  = selectdim(μ,  nd + 1, α)
                dη_α = selectdim(dη, nd + 1, α)

                # Laplacians — initialise dη with ∇²η + ∇²μ                
                cDFT.convolve!(dη_α, η_α, map_lapl, P, iP, buf)
                cDFT.convolve!(tmp,  μ_α, map_lapl, P, iP, buf)
                @. dη_α += tmp

                # Gradient terms — accumulate directly
                for d in 1:nd
                    cDFT.convolve!(tmp, η_α, selectdim(map_grad, nd+1, d), P, iP, buf)   # tmp = ∇ηd
                    @. dη_α += tmp^2                                # |∇η|²

                    cDFT.convolve!(buf_real, μ_α, selectdim(map_grad, nd+1, d), P, iP, buf)  # buf_real = ∇μd
                    @. dη_α += buf_real * tmp                          # ∇μ·∇η
                end
            end
        end

        function total_rhs!(dρ, ρ, params, t)
            # println(t, " ", minimum(ρ), " ", maximum(ρ))

            # system, _, μ, cache_model, cache_external_field, cache_propagator, cache_ddft = params
            # println(t, " ", minimum(ρ), " ", maximum(ρ))
            clamp!(ρ, 1e-8, 1e8)   # prevent overflow in log or exp
            cDFT.δFδρ_res_newautodiff!(system, ρ, μ, cache_model...)
            cDFT.evaluate_external_field!(system, ρ, μ, cache_external_field)
            cDFT.propagate!(system, ρ, μ, cache_propagator)

            for α in axes(ρ, nd+1)
                ρ_α  = selectdim(ρ,  nd+1, α)
                dρ_α = selectdim(dρ, nd+1, α)
                cDFT.convolve!(dρ_α, ρ_α, map_lapl, P, iP, buf)   # ∇²ρ

                # ρ_α  = selectdim(ρ,  nd+1, α)
                μ_α  = selectdim(μ,  nd+1, α)
                # dρ_α = selectdim(dρ, nd+1, α)

                # ∇²μ term — from expanding ∇·(ρ∇μ) = ρ∇²μ + ∇ρ·∇μ
                cDFT.convolve!(tmp, μ_α, map_lapl, P, iP, buf)
                @. dρ_α += ρ_α * tmp                                # ρ∇²μ

                # ∇ρ·∇μ term
                for d in 1:nd
                    cDFT.convolve!(tmp,      ρ_α, selectdim(map_grad, nd+1, d), P, iP, buf)   # ∇ρ
                    cDFT.convolve!(buf_real, μ_α, selectdim(map_grad, nd+1, d), P, iP, buf)   # ∇μ
                    @. dρ_α += tmp * buf_real                      # ∇ρ·∇μ
                end
            end
        end
        return SciMLBase.ODEProblem(ddft_rhs_log!, log.(ρ), tspan, system, kwargs...)
    end

    ######### ALTERNATE CONVERGE! IMPLEMENTATION FOR DDFT STEADY STATES #########
    # function cDFT.converge!(system::cDFT.AbstractcDFTSystem, ρ, alg::SciMLBase.AbstractODEAlgorithm)
    #     cb = TerminateSteadyState(abstol=1e-4, reltol=1e-4)
    #     prob = SciMLBase.ODEProblem(system, log.(ρ), (0.0, 1e6), callback=cb)
    #     sol = DifferentialEquations.solve(prob, alg, save_everystep=false, save_start=false)
    #     ρ .= exp.(sol[end])
    # end
end