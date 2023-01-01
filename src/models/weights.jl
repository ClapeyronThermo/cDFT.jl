function weights_hs(model::SAFTModel,ρ,z,lim)
    bounds = ρ.bounds
    mesh_size = ρ.mesh_size

    upper = bounds[2]

    span = range(-lim,lim,length=41)

    n = ∫ρdz.(Ref(ρ),z,Ref(span))*N_A
    n₃ = ∫ρz²dz.(Ref(ρ),z,Ref(span))*N_A*π
    nᵥ = ∫ρzdz.(Ref(ρ),z,Ref(span))*N_A
    return n, n₃, nᵥ
end

export weights_hs