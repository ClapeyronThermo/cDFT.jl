function adsorption(system,ρ)
    # Integrate over all profiles
    nc = length(system.model)
    nd = dimension(system)
    dz = structure_dz(system.structure)
    if nd == 1
        V = prod(diff(system.structure.bounds))
    else
        V = prod(diff(system.structure.bounds; dims=2))
    end
    return [∫(selectdim(ρ,nd+1,i)[:],dz)/V for i in 1:nc]
end

function adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0])
    L = cDFT.length_scale(model)

    width = surface.params.width[1]
    bounds = [0.7L,width-0.7L]

    v = volume(model,p,T,n)
    ρ = n./v

    structure = cDFT.Uniform1DCart((p, T), ρ, bounds, (201,))

    system = cDFT.DFTSystem(model, structure, surface)

    ρ = initialize_profiles(system)

    converge!(system,ρ)

    return adsorption(system, ρ)
end

export adsorption