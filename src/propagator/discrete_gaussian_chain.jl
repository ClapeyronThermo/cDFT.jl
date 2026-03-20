"""
    DiscreteGaussianChainPropagator(b, N, segment_species, structure, device)

Construct a `DiscreteGaussianChainPropagator` for linear polymer chains.

# Arguments
- `b::Vector{Float64}`: Statistical segment length for each chain.
- `N::Vector{Int}`: Number of segments for each chain.
- `segment_species::Vector{Vector{Int}}`: Mapping from segment index to species index for each chain.
- `structure::DFTStructure`: The spatial discretization structure.
- `device::Backend`: Computation backend (CPU or GPU).
"""
function DiscreteGaussianChainPropagator(
    b::Vector{Float64},
    N::Vector{Int},
    segment_species::Vector{Vector{Int}},
    structure::DFTStructure,
    device::Backend
)
    nchains = length(b)
    @assert length(N) == nchains
    @assert length(segment_species) == nchains
    for c in 1:nchains
        @assert length(segment_species[c]) == N[c]
    end

    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)

    # Build |ν|² grid from ordinary frequencies
    # Each ω̂[i] is a vector of length ngrid[i]; reshape for broadcasting
    ν_sq = zeros(ngrid...)
    for i in 1:nd
        shape = ntuple(d -> d == i ? ngrid[d] : 1, nd)
        ν_sq .+= reshape(ω̂[i] .^ 2, shape)
    end

    # Compute Gaussian transition kernel in Fourier space for each chain
    kernels = Vector{Array{ComplexF64}}(undef, nchains)
    for c in 1:nchains
        # P̂(ν) = exp(-2π²b²|ν|²/3)  — normalized so P̂(0) = 1
        kernels[c] = ComplexF64.(exp.(-2π^2 * b[c]^2 .* ν_sq ./ 3))
    end

    # Move kernels to device
    device_kernels = [Adapt.adapt(device, k) for k in kernels]

    return DiscreteGaussianChainPropagator(device_kernels, b, N, segment_species)
end

function preallocate_propagator(system, propagator::DiscreteGaussianChainPropagator, ρ, backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nchains = length(propagator.N)

    # Allocate forward and backward propagator arrays per chain
    q_fwd = Vector{Any}(undef, nchains)
    q_bwd = Vector{Any}(undef, nchains)
    for c in 1:nchains
        q_fwd[c] = allocate(backend, Float64, ngrid..., propagator.N[c])
        q_bwd[c] = allocate(backend, Float64, ngrid..., propagator.N[c])
    end

    # Shared FFT buffer and plans — use the spatial grid shape
    buf = allocate(backend, ComplexF64, ngrid...)

    if backend isa CPU
        plan = plan_fft!(buf, 1:length(ngrid); num_threads=Threads.nthreads())
    else
        plan = plan_fft!(buf, 1:length(ngrid))
    end

    return q_fwd, q_bwd, buf, plan, inv(plan)
end

function propagate!(system, propagator::DiscreteGaussianChainPropagator, ρ, δfδρ_res, q_fwd, q_bwd, buf, P, iP)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nchains = length(propagator.N)

    for c in 1:nchains
        Nc = propagator.N[c]
        seg_spec = propagator.segment_species[c]
        kernel = propagator.kernels[c]

        # Forward propagator: q(r, i) for i = 1..N (1-indexed)
        # q(r, 1) = exp(-w_{α(1)}(r))
        α1 = seg_spec[1]
        selectdim(q_fwd[c], nd+1, 1) .= exp.(.-selectdim(δfδρ_res, nd+1, α1))

        for i in 2:Nc
            αi = seg_spec[i]
            # Convolve q_fwd(:, i-1) with kernel, store in q_fwd(:, i)
            convolve!(selectdim(q_fwd[c], nd+1, i), selectdim(q_fwd[c], nd+1, i-1), kernel, P, iP, buf)
            # Multiply by exp(-w_{α(i)})
            selectdim(q_fwd[c], nd+1, i) .*= exp.(.-selectdim(δfδρ_res, nd+1, αi))
        end

        # Backward propagator: q†(r, i) for i = 1..N (1-indexed)
        # q†(r, 1) = exp(-w_{α(N)}(r))
        αN = seg_spec[Nc]
        selectdim(q_bwd[c], nd+1, 1) .= exp.(.-selectdim(δfδρ_res, nd+1, αN))

        for i in 2:Nc
            αi = seg_spec[Nc - i + 1]
            # Convolve q_bwd(:, i-1) with kernel, store in q_bwd(:, i)
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf)
            # Multiply by exp(-w_{α(N-i+1)})
            selectdim(q_bwd[c], nd+1, i) .*= exp.(.-selectdim(δfδρ_res, nd+1, αi))
        end

        # Density contribution: modify δfδρ_res
        # δfδρ_res[α] -= log(Σ_{i: α(i)=α} q_fwd(r,i) · q_bwd(r, N+1-i))
        # Determine unique species in this chain
        unique_species = unique(seg_spec)
        for α in unique_species
            # Sum q_fwd(r,i) * q_bwd(r, N+1-i) for all segments i where α(i) = α
            seg_indices = findall(==(α), seg_spec)
            # Accumulate the sum into a temporary (reuse first matching pair, then add)
            first_idx = seg_indices[1]
            # Use broadcasting to compute the sum
            # selectdim(q_fwd[c], nd+1, i) .* selectdim(q_bwd[c], nd+1, Nc+1-i)
            # We need a temporary for the sum; use buf (real part)
            sum_qq = selectdim(q_fwd[c], nd+1, first_idx) .* selectdim(q_bwd[c], nd+1, Nc + 1 - first_idx)
            for idx in seg_indices[2:end]
                sum_qq = sum_qq .+ selectdim(q_fwd[c], nd+1, idx) .* selectdim(q_bwd[c], nd+1, Nc + 1 - idx)
            end
            selectdim(δfδρ_res, nd+1, α) .-= log.(sum_qq)
        end
    end
end
