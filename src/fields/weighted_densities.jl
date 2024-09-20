"""
    WeightedDensity(type::Symbol,width::Vector{Float64})

Generic `WeightedDensity` type used to calculate the weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
"""
struct WeightedDensity <: DFTField 
    type::Symbol
    width::Vector{Float64}
    profiles::Vector{DFTProfile}
end

function WeightedDensity(type::Symbol,width::Vector{Float64},species::DFTSpecies,profiles::Vector{T}) where {T<:DFTProfile}
    ∂f = DensityProfile[]

    nb = length(width)
    δf = zeros(length(profiles[1].coords))
    z = profiles[1].coords

    for i in 1:nb
        lim = species.size[i]
        bounds = profiles[i].bounds.+(-lim,lim)
        bc1 = FreeBoundary(δf[1],-1)
        bc2 = FreeBoundary(δf[end],1)
        push!(∂f, DensityProfile(δf,z,bounds,(bc1,bc2)))
    end
    return WeightedDensity(type,width,∂f)
end

function evaluate_field(n::Matrix{Float64}, structure::DFTStructure,field::WeightedDensity,profiles::Vector{DFTProfile}, species::DFTSpecies)
    nb = length(profiles)
    width = field.width
    size = species.size

    if field.type == :∫ρdz
        integral_method = ∫ρdz
    elseif field.type == :∫ρzdz
        integral_method = ∫ρzdz
    elseif field.type == :∫ρz²dz
        integral_method = ∫ρz²dz
    elseif field.type == :ρ
        for i in 1:length(profiles)
            _profile = profiles[i]
            for j in @grid 
                n[j,i] = _profile.density[j]
                n[j,i] *= N_A
            end
        end
        return n
    else
        error("Invalid type of field")
    end
    
    z = profiles[1].coords

    for i in 1:nb
        span = width[i]*size[i]
        _profile = profiles[i]

        for j in @grid
            n[j,i] = integral_method(structure,
                                     _profile,
                                     z[j],span)
            n[j,i] *= N_A
        end
    end
    return n
end

function integrate_field(δf::Matrix{Float64},structure::DFTStructure,field::WeightedDensity,profiles::Vector{DFTProfile}, species::DFTSpecies)
    nb = length(profiles)

    width = field.width 
    size = species.size
    type = field.type

    z = field.profiles[1].coords

    if type == :∫ρdz
        integral_method = ∫ρdz
        prefactor = 1
    elseif type == :∫ρzdz
        integral_method = ∫ρzdz
        prefactor = -1
    elseif type == :∫ρz²dz
        integral_method = ∫ρz²dz
        prefactor = 1
    elseif type == :ρ
        for i in 1:nb
            δf[:,i] = field.profiles[i].(z)
        end
        return δf
    else
        error("Invalid type of field")
    end

    for i in 1:nb
        span = width[i]*size[i]

        Threads.@threads for j in @grid
            δf[j,i] = prefactor*integral_method(structure,field.profiles[i],z[j],span)
        end
    end
    return δf
end