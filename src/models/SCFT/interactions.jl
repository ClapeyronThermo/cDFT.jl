"""
    compute_fields!(system::SCFTSystem, ρ, w)

Compute the mean-field potential fields `w` from the density profiles `ρ`.
Dispatches to the interaction-model-specific method.
"""
function compute_fields!(system::SCFTSystem, ρ, w)
    compute_fields!(system.interaction, ρ, w, system)
end

"""
    compute_fields!(fh::FloryHuggins, ρ, w, system)

Compute fields for local Flory-Huggins interactions with Helfand compressibility:

    w_α(r) = Σ_β (χ_αβ / ρ₀) ρ_β(r) + (ζ / ρ₀)(ρ₊(r) / ρ₀ - 1)

where ρ₊ = Σ_α ρ_α is the total density.
"""
function compute_fields!(fh::FloryHuggins, ρ, w, system::SCFTSystem)
    nd = dimension(system)
    nspecies = system.nspecies
    ngrid = system.structure.ngrid
    chi = fh.chi
    rho0 = fh.rho0
    kappa = fh.kappa

    # Compute total density ρ₊
    ρ_total = zeros(ngrid...)
    for α in 1:nspecies
        ρ_total .+= selectdim(ρ, nd+1, α)
    end

    # Compressibility term: (ζ / ρ₀)(ρ₊ / ρ₀ - 1)
    comp_term = (kappa / rho0) .* (ρ_total ./ rho0 .- 1.0)

    # Field for each species
    for α in 1:nspecies
        w_α = selectdim(w, nd+1, α)
        w_α .= comp_term
        for β in 1:nspecies
            if chi[α, β] != 0.0
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
function compute_bulk_fields(fh::FloryHuggins, bulk_densities::Vector{Float64})
    nspecies = length(bulk_densities)
    chi = fh.chi
    rho0 = fh.rho0
    kappa = fh.kappa

    ρ_total = sum(bulk_densities)
    comp_term = (kappa / rho0) * (ρ_total / rho0 - 1.0)

    w_bulk = zeros(nspecies)
    for α in 1:nspecies
        w_bulk[α] = comp_term
        for β in 1:nspecies
            w_bulk[α] += (chi[α, β] / rho0) * bulk_densities[β]
        end
    end
    return w_bulk
end
