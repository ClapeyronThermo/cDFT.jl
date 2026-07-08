"""
    DiscreteGaussianChainPropagator(model, species, structure, device, FP=Float64)

Construct a `DiscreteGaussianChainPropagator` for linear polymer chains with
per-species-type segment lengths and automatic junction kernels, matching the generic
`(model, species, structure, device, FP)` constructor signature every other
`DFTPropagator` uses (e.g. `TangentHSPropagator`).

# Arguments
- `model::EoSModel`: Supplies `b_species = model.params.b.values`, the statistical segment
  length for each species type.
- `species::DFTSpecies`: Supplies `species.sequence`, the segment-to-species mapping per
  chain (also gives `N = length.(species.sequence)`). Only used here to build the kernel
  map — not stored on the returned propagator (see its docstring).
- `structure::DFTStructure`: The spatial discretization structure.
- `device::Backend`: Device backend.
- `FP::Type{<:AbstractFloat}`: Float precision (default `Float64`).

For same-species bonds (α, α), the kernel uses `b_α`. For junction bonds (α, β),
the kernel uses `b_αβ = √((b_α² + b_β²) / 2)`.
"""
function DiscreteGaussianChainPropagator(
    model::EoSModel,
    species::DFTSpecies,
    structure::DFTStructure,
    device::Backend,
    ::Type{FP}=Float64
) where FP<:AbstractFloat
    b_species = model.params.b.values
    segment_species = species.sequence
    N = length.(segment_species)
    nchains = length(N)
    @assert length(segment_species) == nchains
    for c in 1:nchains
        @assert length(segment_species[c]) == N[c]
    end

    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)

    # Build |ν|² on the R2C (half-complex) grid.
    # First dimension uses rfftfreq (0..N/2 only), rest use fftfreq.
    # This matches the output layout of plan_rfft, halving the kernel memory and
    # matching the complexity of a real-to-complex FFT (~2× cheaper than C2C).
    lb1, ub1 = bounds(structure, 1)
    ω̂1_rfft  = rfftfreq(ngrid[1], ngrid[1] / (ub1 - lb1))
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)

    ν_sq = zeros(rfft_ngrid...)
    # Dimension 1: rfftfreq
    ν_sq .+= reshape(ω̂1_rfft .^ 2, ntuple(d -> d == 1 ? rfft_ngrid[1] : 1, nd))
    # Dimensions 2..nd: standard fftfreq
    for i in 2:nd
        ν_sq .+= reshape(ω̂[i] .^ 2, ntuple(d -> d == i ? rfft_ngrid[d] : 1, nd))
    end

    # Find all unique adjacent species pairs across all chains
    bond_pairs = Set{Tuple{Int,Int}}()
    for c in 1:nchains
        seg_spec = segment_species[c]
        for i in 2:N[c]
            push!(bond_pairs, minmax(seg_spec[i-1], seg_spec[i]))
        end
    end

    # Compute kernel for each unique bond pair in R2C (half-complex) shape.
    # Precision is controlled by FP (set via DFTOptions' precision kwarg; Float64 by default).
    CT = Complex{FP}
    kernel_map = Dict{Tuple{Int,Int}, Array{CT}}()
    for (α, β) in bond_pairs
        if α == β
            b_bond = b_species[α]
        else
            b_bond = sqrt((b_species[α]^2 + b_species[β]^2) / 2)
        end
        kernel_map[(α, β)] = CT.(exp.(-2π^2 * b_bond^2 .* ν_sq ./ 3))
    end

    # Move kernels to device.
    # Use a dummy CPU array of the right shape/type to determine the device array type,
    # which avoids calling first() on an empty dict when nchains == 0 (solvent-only system).
    dummy_cpu = CT.(zeros(rfft_ngrid...))
    device_kernel_map = Dict{Tuple{Int,Int}, typeof(Adapt.adapt(device, dummy_cpu))}()
    for (key, k) in kernel_map
        device_kernel_map[key] = Adapt.adapt(device, k)
    end

    return DiscreteGaussianChainPropagator(device_kernel_map)
end

function preallocate_propagator(system, propagator::DiscreteGaussianChainPropagator, ρ, backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    sequence = system.species.sequence
    nchains = length(sequence)

    # Allocate forward and backward propagator arrays per chain.
    # Use a concrete element type (not Vector{Any}) to avoid type-dispatch overhead
    # in the hot propagation loop. When nchains == 0 (solvent-only system), allocate
    # empty typed vectors using a dummy size so downstream code has a concrete type.
    FP = fptype(system.options)
    CT = Complex{FP}
    proto_N = nchains > 0 ? length(sequence[1]) : 1
    q_proto = allocate(backend, FP, ngrid..., proto_N)
    q_fwd   = Vector{typeof(q_proto)}(undef, nchains)
    q_bwd   = Vector{typeof(q_proto)}(undef, nchains)
    if nchains > 0
        q_fwd[1] = q_proto
        q_bwd[1] = allocate(backend, FP, ngrid..., length(sequence[1]))
        for c in 2:nchains
            q_fwd[c] = allocate(backend, FP, ngrid..., length(sequence[c]))
            q_bwd[c] = allocate(backend, FP, ngrid..., length(sequence[c]))
        end
    end

    # R2C (real-to-complex) FFT buffers:
    #   buf_r — real input,  shape ngrid
    #   buf_c — complex output, shape (ngrid[1]÷2+1, ngrid[2:end]...)
    # Using R2C instead of C2C halves the FFT work (~2× faster for real data).
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    buf_r = allocate(backend, FP, ngrid...)
    buf_c = allocate(backend, CT, rfft_ngrid...)

    if backend isa CPU
        P  = plan_rfft(buf_r,  1:nd; num_threads=Threads.nthreads())
        iP = plan_irfft(buf_c, ngrid[1], 1:nd; num_threads=Threads.nthreads())
    else
        P  = plan_rfft(buf_r,  1:nd)
        iP = plan_irfft(buf_c, ngrid[1], 1:nd)
    end

    return q_fwd, q_bwd, buf_r, buf_c, P, iP
end

function propagate!(system, propagator::DiscreteGaussianChainPropagator, ρ, δfδρ_res, q_fwd, q_bwd, buf_r, buf_c, P, iP)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    sequence = system.species.sequence
    nchains = length(sequence)

    for c in 1:nchains
        seg_spec = sequence[c]
        Nc = length(seg_spec)

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
            convolve!(selectdim(q_fwd[c], nd+1, i), selectdim(q_fwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
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
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
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

"""
    propagate!(system::SCFTSystem, ρ, w, cache_propagator; w_bulk, exp_field=nothing)

Run discrete-Gaussian-chain forward/backward propagator sweeps using shifted fields
`Δw = w - w_bulk`. `ρ` is accepted but unused — kept purely for positional uniformity
with every other `propagate!` method (`IdealPropagator`, `TangentHSPropagator`), all of
which take `(system, ρ, field, cache_propagator)`.

Shifting by `w_bulk` keeps propagator values near O(1) for near-uniform systems,
avoiding the numerical underflow that occurs when raw fields are large.
The shifted propagator satisfies `q̃(r,s) = q(r,s) * exp(Σ_{t=1}^{s} w_bulk[α(t)])`,
so `Q̃ ≈ 1` for uniform systems (instead of `Q ∼ exp(-N * w_bulk) ≈ 0`).
"""
function propagate!(system::SCFTSystem, ρ, w, cache_propagator;
                    w_bulk, exp_field=nothing)
    q_fwd, q_bwd, buf_r, buf_c, P, iP = cache_propagator
    nd = dimension(system)
    propagator = system.propagator
    sequence = system.species.sequence
    nchains = length(sequence)

    # Helper: return precomputed exp_field[α] if available, else compute on the fly.
    # exp_field[α] = exp(w_bulk[α] - w_α(r))
    ef(α) = exp_field !== nothing ? exp_field[α] :
                exp.(w_bulk[α] .- selectdim(w, nd+1, α))

    for c in 1:nchains
        seg_spec = sequence[c]
        Nc = length(seg_spec)

        # Forward propagator with shifted fields
        α1 = seg_spec[1]
        selectdim(q_fwd[c], nd+1, 1) .= ef(α1)

        # Backward propagator with shifted fields
        αN = seg_spec[Nc]
        selectdim(q_bwd[c], nd+1, 1) .= ef(αN)

        for i in 2:Nc
            αi = seg_spec[i]
            bond_key = minmax(seg_spec[i-1], seg_spec[i])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_fwd[c], nd+1, i), selectdim(q_fwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
            selectdim(q_fwd[c], nd+1, i) .*= ef(αi)
        end

        for i in 2:Nc
            αi = seg_spec[Nc - i + 1]
            bond_key = minmax(seg_spec[Nc - i + 1], seg_spec[Nc - i + 2])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
            selectdim(q_bwd[c], nd+1, i) .*= ef(αi)
        end
    end
end
