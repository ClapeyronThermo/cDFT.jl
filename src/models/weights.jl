function weights_hs(model::SAFTModel,ρ,z,lim)
    bounds = ρ.bounds
    mesh_size = ρ.mesh_size

    upper = bounds[2]

    z_damp = 0:mesh_size:(upper+lim)
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    m = model.params.segment.values[1]

    span = range(-lim,lim,length=101)

    n = ∫ρdz.(Ref(ρ),z_damp,Ref(span))*m*N_A
    n₃ = ∫ρz²dz.(Ref(ρ),z_damp,Ref(span))*m*N_A*π
    nᵥ = ∫ρzdz.(Ref(ρ),z_damp,Ref(span))*m*N_A
    return n, n₃, nᵥ
end

export weights_hs