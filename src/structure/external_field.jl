include("external_fields/steele.jl")

function evaluate_external_field(system::DFTSystem)
    structure = system.structure
    
    if !hasfield(typeof(structure),:external_field)
        ngrid = structure.ngrid
        nbeads = length(system.profiles)
        return zeros(ngrid,nbeads)
    end

    model = system.model
    external_field = structure.external_field
    profiles = system.profiles
    return evaluate_external_field(structure,external_field,model,profiles)
end

function initialize_profiles(model::EoSModel,structure::ExternalField1DCart, species)
    external_field = structure.external_field
    H = structure.width
    nc = length(model)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, n) = structure.conditions
    bounds = structure.bounds
    z_interface = sum(bounds)/2

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    v = volume(model,p,T,n)

    Vext = evaluate_external_field(structure,external_field,model,z)
    zmax = z[[argmax(exp.(-Vext[:,i])) for i in 1:size(Vext,2)]]

    coef = 1/(0.1L)

    ρ = DensityProfile[]
    for i in @comps
        for j in 1:species.nbeads[i]
            if H == 0
                ρb = n[i]/v
                _ρ = tanh_prof.(z,ρb,0.,zmax[i],coef).*exp.(-Vext[:,i]/10)

                ρ1 = ρb
                ρ2 = _ρ[1]
            else
                ρb = p/(R̄*T)*n[i]
                _ρ = sqrt.(tanh_prof.(z,ρb,0.,zmax[i],coef).*tanh_prof.(z,0.,ρb,H - zmax[i],coef)).*exp.(-Vext[:,i]/10)

                ρ1 = _ρ[end]
                ρ2 = _ρ[1]
            end
            boundary_conditions = (FreeBoundary(ρ1,-1),FreeBoundary(ρ2,1))
            push!(ρ,DensityProfile(_ρ,z,bounds,boundary_conditions))
        end
    end

    return ρ
end

