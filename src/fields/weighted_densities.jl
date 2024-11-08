"""
    WeightedDensity(type::Symbol,width::Vector{Float64})

Generic `WeightedDensity` type used to calculate the weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
"""
struct WeightedDensity <: DFTField 
    type::Symbol
    width::Vector{Float64}
    map::Array{ComplexF64}
end

function WeightedDensity(type::Symbol,width::Vector{Float64},ω::Frequencies{Float64})
    
    R = 2π.*width'

    if type == :∫ρdz
        Ω = 2*R .* (ω .== 0.0) + 2*sin.(ω.*R)./ω .*(ω .!= 0.0)
        Ω ./= 2π
    elseif type == :∫ρzdz
        Ω = 4π*im./ω.^2 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) .+ 0.0
        Ω ./= (2π)^3
    elseif type == :∫ρz²dz
        Ω = 4π./ω.^3 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) + R.^3/3*4π .*(ω .== 0.0)
        Ω ./= (2π)^3
    elseif type == :ρ
        Ω = ones(length(ω),length(width))
    else
        error("Invalid type of field")
    end
    return WeightedDensity(type,width,Ω)
end

function evaluate_field(system::DFTSystem,field::WeightedDensity, ρ)
    structure = system.structure
    nb = size(ρ,2)
    ngrid = structure.ngrid
    n = zeros(ngrid,nb)

    map = field.map
    if field.type == :ρ
        return ρ.*N_A
    end

    for i in 1:nb
        n[:,i] = real.(ifft(fft(ρ[:,i]).*map[:,i]))*N_A
    end
    return n
end

function integrate_field(system::DFTSystem,field::WeightedDensity,profile)
    map = field.map 
    type = field.type
    nb = size(profile,2)
    ngrid = system.structure.ngrid

    if type == :∫ρdz || type == :∫ρz²dz
        prefactor = 1
    elseif type == :∫ρzdz
        prefactor = -1
    elseif type == :ρ
        return profile
    else
        error("Invalid type of field")
    end

    ∫field = zeros(ngrid,nb)

    for i in 1:nb
        ∫field[:,i] = prefactor*real.(ifft(fft(profile[:,i]).*map[:,i]))
    end
    return ∫field
end