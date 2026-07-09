preallocate_external_potential(system::DGTSystem, ρ) = nothing
preallocate_propagator(system::DGTSystem, ρ)         = nothing

struct DGTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
Pointwise residual free energy for DGT, written in Enzyme-compatible style.
`n[kk,1,i]` = smoothed density ρ̄ᵢ; `n[kk,1+d,i]` = gradient component d of ρ̄ᵢ.
The bulk EoS term calls Clapeyron.a_res with the model stored in Const(params);
the gradient correction is inlined arithmetic over the κ matrix.
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: DGTSystem}
    κ = params.kappa

    ∑ρ = zero(eltype(n))
    for i in 1:NC
        ∑ρ += n[kk, 1, i]
    end

    z = Vector{Float64}(undef, NC)
    for i in 1:NC
        z[i] = n[kk, 1, i] / N_A
    end
    bulk_term = a_res(params.model, 1.0, T, z) * ∑ρ

    grad_term = zero(eltype(n))
    for i in 1:NC
        for d in 1:ND
            grad_term += κ[i, i] * n[kk, 1+d, i]^2 / 2
        end
        for j in i+1:NC
            for d in 1:ND
                grad_term += κ[i, j] * n[kk, 1+d, i] * n[kk, 1+d, j]
            end
        end
    end

    out[kk] = bulk_term + grad_term / T
    return nothing
end

include("gradients/const.jl")

function preallocate_params(system::DGTSystem)
    T  = system.structure.conditions[2]
    ρ̄  = system.species.bulk_density
    κ  = kappa(system.gradient, T, ρ̄)
    return (; kappa = κ, model = system.model), length(system.model)
end

function preallocate_model(system::DGTSystem, ρ)
    nf    = length_fields(system)
    ngrid = system.structure.ngrid
    nd    = length(ngrid)
    nb    = size(ρ, nd+1)

    n       = allocate(CPU(), Float64, ngrid..., nf, nb)
    δf      = allocate(CPU(), Float64, ngrid..., nf, nb);  fill!(δf, 0.0)
    fft_buf = allocate(CPU(), Float64, ngrid..., nf, nb)
    in_buf  = allocate(CPU(), ComplexF64, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan    = plan_fft!(tmp, 1:nd; num_threads = Threads.nthreads())
    iplan   = inv(plan)
    f_val   = allocate(CPU(), Float64, ngrid...)
    δf_val  = allocate(CPU(), Float64, ngrid...);  fill!(δf_val, 1.0)

    params, nc = preallocate_params(system)

    # Mirrors preallocate_model(::DFTSystem,...)/(::ElectrolyteDFTSystem,...)
    # (src/models/models.jl) — δFδρ_res! destructures fwd_cache unconditionally whenever
    # system.options.ad_mode === :forward (the DFTOptions default), so DGT needs the same
    # dn_seeds/df_outs/batch construction, not just `nothing`.
    if system.options.ad_mode === :forward
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(CPU(), Float64, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(CPU(), Float64, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end
