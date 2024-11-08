function adsorption(system,ρ)
    # Integrate over all profiles
    return [∫(_ρ,_ρ.mesh_size)/diff(_ρ.bounds)[1] for _ρ in profiles]
end

function adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0], width=0.0)
    L = cDFT.length_scale(model)

    if width == 0.
        bounds = [0.7L,10L]
    else
        bounds = [0.7L,width-0.7L]
    end
    structure = cDFT.ExternalField1DCart((p, T, n),bounds, 201, surface, width)

    system = cDFT.DFTSystem(model, structure)

    converge!(system)

    return adsorption(system)
end