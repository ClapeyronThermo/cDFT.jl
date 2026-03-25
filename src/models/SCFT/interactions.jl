"""
    compute_fields!(system::SCFTSystem, ρ, w)

Compute the mean-field potential fields `w` from the density profiles `ρ`.
Dispatches to the interaction-model-specific method.
"""
function compute_fields!(system::SCFTSystem, ρ, w; scratch=nothing)
    compute_fields!(system.interaction, ρ, w, system; scratch=scratch)
end

"""
    compute_fields!(fh::FloryHuggins, ρ, w, system)

Compute fields for local Flory-Huggins interactions with Helfand compressibility:

    w_α(r) = Σ_β (χ_αβ / ρ₀) ρ_β(r) + (ζ / ρ₀)(ρ₊(r) / ρ₀ - 1)

where ρ₊ = Σ_α ρ_α is the total density.
"""
function compute_fields!(fh::FloryHuggins, ρ, w, system::SCFTSystem; scratch=nothing)
    nd = dimension(system)
    nspecies = system.nspecies
    FT = eltype(w)
    chi = FT.(fh.chi)
    rho0 = FT(fh.rho0)
    kappa = FT(fh.kappa)

    # Use preallocated scratch if provided, otherwise allocate.
    # scratch must have the same shape/device as a single species slice of w.
    ρ_total = scratch !== nothing ? scratch : similar(selectdim(w, nd+1, 1))

    # Accumulate total density into scratch (in-place, no allocation)
    ρ_total .= zero(FT)
    for α in 1:nspecies
        ρ_total .+= selectdim(ρ, nd+1, α)
    end

    # Overwrite ρ_total in-place with the compressibility term to avoid a second allocation:
    # comp_term = (ζ / ρ₀)(ρ₊ / ρ₀ - 1)
    @. ρ_total = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    # Field for each species
    for α in 1:nspecies
        w_α = selectdim(w, nd+1, α)
        w_α .= ρ_total
        for β in 1:nspecies
            if chi[α, β] != zero(FT)
                w_α .+= (chi[α, β] / rho0) .* selectdim(ρ, nd+1, β)
            end
        end
    end
end

"""
    compute_bulk_fields(fh::FloryHuggins, bulk_densities::Vector{Float64})

Compute bulk (uniform) fields from bulk densities. Returns a vector of field
values, one per species.
"""
function compute_bulk_fields(fh::FloryHuggins, bulk_densities::AbstractVector{T}) where {T<:AbstractFloat}
    nspecies = length(bulk_densities)
    FT = eltype(bulk_densities)
    chi = FT.(fh.chi)
    rho0 = FT(fh.rho0)
    kappa = FT(fh.kappa)

    ρ_total = sum(bulk_densities)
    comp_term = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    w_bulk = zeros(FT, nspecies)
    for α in 1:nspecies
        w_bulk[α] = comp_term
        for β in 1:nspecies
            w_bulk[α] += (chi[α, β] / rho0) * bulk_densities[β]
        end
    end
    return w_bulk
end
