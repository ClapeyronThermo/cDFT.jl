struct DGTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function f_res(system::DGTSystem, model::EoSModel,n)
    nd = dimension(system)
    n1,n2 = @view(n[1,:]),@view(n[2:2+nd-1,:])
    (p,T) = system.structure.conditions
    ∑ρ = sum(n1)
    return a_res(model, 1., T, n1./N_A)*∑ρ + ∇a_res(system,system.gradient,n1,n2)
end

function ∇a_res(system::DGTSystem,gradient::GradientModel, ρ̄, ∇ρ̄)
    (p,T) = system.structure.conditions
    κ = kappa(gradient, T, ρ̄)
    _res = zero(eltype(ρ̄))
    for i in 1:length(ρ̄)
        ∇ρ̄i = @view ∇ρ̄[:,i]
        _res += κ[i,i]*dot(∇ρ̄i,∇ρ̄i)/2
        for j in i+1:length(ρ̄)
            ∇ρ̄j = @view ∇ρ̄[:,j]
            _res += κ[i,j]*dot(∇ρ̄i,∇ρ̄j)
        end
    end
    return _res/T
end

include("gradients/const.jl")

function preallocate_model(system::DGTSystem, ρ)
    backend = system.options.device
    nf = length_fields(system)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ, nd+1)

    n       = allocate(CPU(), Float64, ngrid..., nf, nb)
    δf      = allocate(CPU(), Float64, ngrid..., nf, nb)
    fft_buf = allocate(backend, Float64, ngrid..., nf, nb)
    in_buf  = allocate(backend, ComplexF64, ngrid...)
    out_buf = similar(in_buf)

    tmp = similar(in_buf)
    plan  = backend isa CPU ?
            plan_fft!(tmp, 1:nd; num_threads=Threads.nthreads()) :
            plan_fft!(tmp, 1:nd)
    iplan = inv(plan)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan
end