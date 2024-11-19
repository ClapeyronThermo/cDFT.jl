struct LJFieldParam <: EoSParam
    sigma::SingleParam{Float64}
    epsilon::SingleParam{Float64}
    position::SingleParam{NTuple{3,Float64}}
end

abstract type LJFieldModel <: ExternalFieldModel end

struct LJField <: LJFieldModel
    surface::Array{String,1}
    params::SteeleParam
    references::Array{String,1}
end

export Steele

function LJField(atoms::Vector{String}, position::Vector{NTuple{3,Float64}}, topology::Array{Float64,1})
    ϵs = SingleParam(topology[:,1])
    σs = SingleParam(topology[:,2].*1e-10)

    position = SingleParam(position)

    packagedparams = LJFieldParam(σs,ϵs,position)
    return LJField(atoms, packagedparams, references)
end

function LJField(position::String,topology::String)
    type = split(position,".")[end]
    if type == "gro"
        pos = readdlm(position, ' '; skipstart=2)
        atoms = pos[:,2]
        position = Float64.(pos[:,4:6])
    else
        error("File type not supported")
    end

    natom = length(atoms)

    topology = zeros(natom,2)
    Mw = zeros(natom)

    type = split(topology,".")[end]
    if type == "itp"
        top = readdlm(topology, ' '; skipstart=1)
        # split top by the atoms section and the atomptypes sections
        top_atomtypes = top[findfirst(x->x=="[ atomtypes ]",top[:,1])+1:findfirst(x->x=="[ atoms ]",top[:,1])-1,:]
        top_atom = top[findfirst(x->x=="[ atoms ]",top[:,1])+1:findfirst(x->x=="[ atoms ]",top[:,1])+natom,:]
        for i in 1:natom
            opls_name = top_atom[findfirst(x->x==atoms[i],top_atom[:,5]),1]
            idx = findfirst(x->x==opls_name,top_atomtypes[:,1])
            topology[i,1] = Float64(top_atomtypes[idx,6])./1e-1
            topology[i,2] = Float64(top_atomtypes[idx,7]).*1e3/Clapeyron.R̄
            Mw[i] = Float64(top_atomtypes[idx,3])
        end
    else
        error("File type not supported")
    end

    # Rescale the coordinates to have the center of mass at 0,0,0

    com = sum(position.*Mw, dims=1)./sum(Mw)
    position .-= com
    position = [Tuple(position[i,:]./1e-1) for i in 1:natom]

    return LJField(atoms, position, topology)
end

function evaluate_external_field(structure::DFTStructure,external_field::SteeleModel,model::SAFTModel,z)
    nd = dimension(structure)
    (_,T) = structure.conditions
    ϵs = external_field.params.epsilon.values
    σs = external_field.params.sigma.values
    zs  = external_field.params.position.values

    ϵi = diagvalues(model.params.epsilon.values)
    σi = diagvalues(model.params.sigma.values)
    
    ngrid = structure.ngrid
    nbeads = length(ϵi)
    nsurf = length(ϵs)
    external_field_values = zeros(ngrid...,nbeads)
    for s in 1:nsurf
        r = sqrt.(sum((zs[s]-z).^2, dims=1))
        ϵsi = sqrt.(ϵs[s].*ϵi)
        σsi = (σs[s].+σi)/2
        for i in 1:nbeads
            selectdim(external_field_values,nd+1,i) .+= @. 4*ϵsi[i]*((σsi[i]/r)^12-(σsi[i]/r)^6)
        end
    end
    return external_field_values./T
end

function evaluate_external_field(structure::DFTStructure,external_field::SteeleModel,model::SAFTModel,ρ::Array{Float64},z)
    return evaluate_external_field(structure,external_field,model,z)
end