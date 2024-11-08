include("external_fields/steele.jl")

function evaluate_external_field(system::DFTSystem, ρ, z)
    structure = system.structure
    
    if !hasfield(typeof(structure),:external_field)
        ngrid = structure.ngrid
        nbeads = length(ρ)
        return zeros(ngrid,nbeads)
    end

    model = system.model
    external_field = structure.external_field
    return evaluate_external_field(structure,external_field,model,ρ,z)
end

function initialize_profiles(model::EoSModel,structure::ExternalField1DCart, species)
    external_field = structure.external_field
    H = structure.width
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    bounds = structure.bounds

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    ρbulk = structure.ρbulk

    Vext = evaluate_external_field(structure,external_field,model,z)
    zmax = z[[argmax(exp.(-Vext[1:Int(round(ngrid/2)),i])) for i in 1:size(Vext,2)]]

    coef = 1/(0.1L)

    ρ = zeros(ngrid,sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            if H == 0
                _ρ = tanh_prof.(z,ρbulk[i],0.,zmax[i],coef).*exp.(-Vext[:,i]/10)

                ρ1 = ρbulk[i]
                ρ2 = _ρ[1]
            else
                ρb = p/(R̄*T)*ρbulk[i]/sum(ρbulk)
                _ρ = sqrt.(tanh_prof.(z,ρbulk,0.,zmax[i],coef).*tanh_prof.(z,0.,ρb,H - zmax[i],coef)).*exp.(-Vext[:,i]/10)

                ρ1 = _ρ[end]
                ρ2 = _ρ[1]
            end
            ρ[:,j] = _ρ
        end
    end

    return ρ
end

