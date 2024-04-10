struct WeightedDensity <: DFTField 
    type::Symbol
    width::Vector{Float64}
end

function weights_hs(structure::DFTStructure,ρ::DFTProfile,field::WeightedDensity)
    n = zeros(length(ρ.z),length(ρ))
    width = field.width 
    type = field.type

    if type == :∫ρdz
        integral_method = ∫ρdz
    elseif type == :∫ρzdz
        integral_method = ∫ρzdz
    elseif type == :∫ρz²dz
        integral_method = ∫ρz²dz
    elseif type == :ρ
        for i in @comps
            n[:,i] .= ρ[i].density
        end
        return n
    else
        error("Invalid type of field")
    end
    
    z = ρ.coords

    for i in @comps
        span = range(-width[i],width[i],length=41)
        
        n[:,i] .= integral_method.(structure,Ref(ρ[i]),z,Ref(span))*N_A
    end
    return n
end