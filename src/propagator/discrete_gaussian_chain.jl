"""
    DiscreteGaussianChainPropagator(b_species, N, segment_species, structure, device)

Construct a `DiscreteGaussianChainPropagator` for linear polymer chains with
per-species-type segment lengths and automatic junction kernels.

# Arguments
- `b_species::Vector{Float64}`: Statistical segment length for each species type.
- `N::Vector{Int}`: Number of segments for each chain.
- `segment_species::Vector{Vector{Int}}`: Mapping from segment index to species index for each chain.
- `structure::DFTStructure`: The spatial discretization structure.
- `device::Backend`: Computation backend (CPU or GPU).

For same-species bonds (α, α), the kernel uses `b_α`. For junction bonds (α, β),
the kernel uses `b_αβ = √((b_α² + b_β²) / 2)`.
"""
function DiscreteGaussianChainPropagator(
    b_species::Vector{Float64},
    N::Vector{Int},
    segment_species::Vector{Vector{Int}},
    structure::DFTStructure,
    device::Backend
)
    nchains = length(N)
    @assert length(segment_species) == nchains
    for c in 1:nchains
        @assert length(segment_species[c]) == N[c]
    end

    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)

    # Build |ν|² grid from ordinary frequencies
    ν_sq = zeros(ngrid...)
    for i in 1:nd
        shape = ntuple(d -> d == i ? ngrid[d] : 1, nd)
        ν_sq .+= reshape(ω̂[i] .^ 2, shape)
    end

    # Find all unique adjacent species pairs across all chains
    bond_pairs = Set{Tuple{Int,Int}}()
    for c in 1:nchains
        seg_spec = segment_species[c]
        for i in 2:N[c]
            push!(bond_pairs, minmax(seg_spec[i-1], seg_spec[i]))
        end
    end

    # Compute kernel for each unique bond pair
    kernel_map = Dict{Tuple{Int,Int}, Array{ComplexF64}}()
    for (α, β) in bond_pairs
        if α == β
            b_bond = b_species[α]
        else
            b_bond = sqrt((b_species[α]^2 + b_species[β]^2) / 2)
        end
        kernel_map[(α, β)] = ComplexF64.(exp.(-2π^2 * b_bond^2 .* ν_sq ./ 3))
    end

    # Move kernels to device
    device_kernel_map = Dict{Tuple{Int,Int}, typeof(Adapt.adapt(device, first(values(kernel_map))))}()
    for (key, k) in kernel_map
        device_kernel_map[key] = Adapt.adapt(device, k)
    end

    return DiscreteGaussianChainPropagator(device_kernel_map, b_species, N, segment_species)
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

        # Forward propagator: q(r, i) for i = 1..N (1-indexed)
        # q(r, 1) = exp(-w_{α(1)}(r))
        α1 = seg_spec[1]
        selectdim(q_fwd[c], nd+1, 1) .= exp.(.-selectdim(δfδρ_res, nd+1, α1))

        for i in 2:Nc
            αi = seg_spec[i]
            # Select kernel based on the bond between segments i-1 and i
            bond_key = minmax(seg_spec[i-1], seg_spec[i])
            kernel = propagator.kernel_map[bond_key]
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
            # Bond between segments (Nc-i+1) and (Nc-i+2) in the original chain
            bond_key = minmax(seg_spec[Nc - i + 1], seg_spec[Nc - i + 2])
            kernel = propagator.kernel_map[bond_key]
            # Convolve q_bwd(:, i-1) with kernel, store in q_bwd(:, i)
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf)
            # Multiply by exp(-w_{α(N-i+1)})
            selectdim(q_bwd[c], nd+1, i) .*= exp.(.-selectdim(δfδρ_res, nd+1, αi))
        end

        # Density contribution: modify δfδρ_res
        unique_species = unique(seg_spec)
        for α in unique_species
            seg_indices = findall(==(α), seg_spec)
            first_idx = seg_indices[1]
            sum_qq = selectdim(q_fwd[c], nd+1, first_idx) .* selectdim(q_bwd[c], nd+1, Nc + 1 - first_idx)
            for idx in seg_indices[2:end]
                sum_qq = sum_qq .+ selectdim(q_fwd[c], nd+1, idx) .* selectdim(q_bwd[c], nd+1, Nc + 1 - idx)
            end
            selectdim(δfδρ_res, nd+1, α) .-= log.(sum_qq)
        end
    end
end
