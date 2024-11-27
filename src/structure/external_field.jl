include("external_fields/steele.jl")
include("external_fields/lj.jl")

function evaluate_external_field(system::DFTSystem, ρ, z)
    structure = system.structure
    
    if !hasfield(typeof(structure),:external_field)
        ngrid = structure.ngrid
        nbeads = sum(system.species.nbeads)
        return zeros(ngrid...,nbeads)
    end

    model = system.model
    external_field = structure.external_field
    return evaluate_external_field(structure,external_field,model,ρ,z)
end

function initialize_profiles(model::EoSModel,structure::ExternalField1DCart, species)
    nd = dimension(structure)
    external_field = structure.external_field
    H = structure.width
    bounds = structure.bounds
    ngrid = structure.ngrid[1]
    (p, T) = structure.conditions
    bounds = structure.bounds

    z = uniform_range(structure,1)
    L = length_scale(model)

    ρbulk = structure.ρbulk

    Vext = evaluate_external_field(structure,external_field,model,z)
    zmax = z[[argmax(exp.(-Vext[1:Int(round(ngrid/2)),i])) for i in 1:size(Vext,2)]]
    coef = 1/(0.1L)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            if H == 0
                _ρ = tanh_prof.(z,ρbulk[i],0.,zmax[i],coef).*exp.(-Vext[:,i]/10)
            else
                ρb = p/(R̄*T)*ρbulk[i]/sum(ρbulk)
                _ρ = ρbulk[i].*exp.(-Vext[:,i]/10)
            end
            selectdim(ρ,nd+1,j) .= _ρ
        end
    end

    return ρ
end

function initialize_profiles(model::EoSModel,structure::ExternalField3DCart, species)
    nd = dimension(structure)
    external_field = structure.external_field
    ngrid = structure.ngrid

    x = uniform_range(structure,1)
    y = uniform_range(structure,2)
    z = uniform_range(structure,3)
    Z = zeros((ngrid...,3))
    for i in 1:ngrid[1]
        Z[i,:,:,1] .= x[i]
    end

    for i in 1:ngrid[2]
        Z[:,i,:,2] .= y[i]
    end

    for i in 1:ngrid[3]
        Z[:,:,i,3] .= z[i]
    end

    ρbulk = structure.ρbulk

    Vext = evaluate_external_field(structure,external_field,model,Z)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            _ρ = ρbulk[i].*exp.(-selectdim(Vext,nd+1,i)/10)
            selectdim(ρ,nd+1,j) .= _ρ
        end
    end

    return ρ
end