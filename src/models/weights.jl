function weights_hs(model::SAFTModel,ρ,z,lim)
    n = zeros(length(z),length(ρ))
    n₃ = zeros(length(z),length(ρ))
    nᵥ = zeros(length(z),length(ρ))
    for i in @comps
        bounds = ρ[i].bounds
        mesh_size = ρ[i].mesh_size

        upper = bounds[2]

        span = range(-lim[i],lim[i],length=41)

        n[:,i] .= ∫ρdz.(Ref(ρ[i]),z,Ref(span))*N_A
        n₃[:,i] .= ∫ρz²dz.(Ref(ρ[i]),z,Ref(span))*N_A*π
        nᵥ[:,i] .= ∫ρzdz.(Ref(ρ[i]),z,Ref(span))*N_A
    end
    return n, n₃, nᵥ
end

export weights_hs