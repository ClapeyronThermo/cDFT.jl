function F_ideal(model::BasicIdealModel,ρ,T,z)
    dz = ρ.mesh_size

    ρ_eval = ρ.(z)
    
    Φ = f_ideal.(Ref(model), Ref(T), ρ_eval)
    return ∫(Φ,dz)
end

function f_ideal(model::BasicIdealModel, T, ρ)
    return @. N_A*ρ*(log(ρ*T^-1.5)-1)
end