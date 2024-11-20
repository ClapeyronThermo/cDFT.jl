struct LJFieldParam <: EoSParam
    sigma::SingleParam{Float64}
    epsilon::SingleParam{Float64}
    position::SingleParam{NTuple{3,Float64}}
end

abstract type LJFieldModel <: ExternalFieldModel end

struct LJField <: LJFieldModel
    surface::Array{String,1}
    params::LJFieldParam
    references::Array{String,1}
end

export LJField

function LJField(atoms::Vector{String}, position::Vector{Tuple{Float64,Float64,Float64}}, topology::Matrix{Float64})
    ϵs = SingleParam("epsilon", atoms, topology[:,2])
    σs = SingleParam("sigma", atoms, topology[:,1].*1e-10)

    position = SingleParam("position", atoms, position)

    packagedparams = LJFieldParam(σs,ϵs,position)
    references = String[]
    return LJField(atoms, packagedparams, references)
end

function LJField(position::String,topology::String)
    type = split(position,".")[end]

    if type == "gro"
        pos = readdlm(position, ' '; skipstart=2)
        # Remove "' from pos
        natoms = size(pos,1)-1
        atoms = Vector{String}(undef,natoms)
        position = zeros(natoms,3)
        for i in 1:natoms
            _pos = pos[i,pos[i,:] .!= ""]
            position[i,:] = _pos[4:6]
            atoms[i] = _pos[2]
        end
    else
        error("File type not supported")
    end


    Mw = zeros(natoms)

    type = split(topology,".")[end]
    if type == "itp"
        top = readdlm(topology, ' '; skipstart=1)
        topology = zeros(natoms,2)

        # split top by the atoms section and the atomptypes sections
        top_atomtypes = top[findfirst(x->x=="atomtypes",top[:,2])+1:findfirst(x->x=="moleculetype",top[:,2])-1,:]
        top_atom = top[findfirst(x->x=="atoms",top[:,2])+2:findfirst(x->x=="atoms",top[:,2])+natoms+1,:]
        opls_name = Vector{String}(undef,natoms)
        top_names = Vector{String}(undef,natoms)
        opls_atomtypes = Vector{String}(undef,natoms)
        for i in 1:natoms
            # remove "" from top_atom
            _top_atom = top_atom[i,top_atom[i,:] .!= ""]
            opls_name[i] = _top_atom[2]
            top_names[i] = _top_atom[5]

            _top_atomtype = top_atomtypes[i,top_atomtypes[i,:] .!= ""]
            opls_atomtypes[i] = _top_atomtype[1]
        end

        for i in 1:natoms
            _opls_name = opls_name[findfirst(x->x==atoms[i],top_names)]
            idx = findfirst(x->x==_opls_name,opls_atomtypes)
            _top_atomtype = top_atomtypes[idx,top_atomtypes[idx,:] .!= ""]
            topology[i,1] = Float64(_top_atomtype[6])./1e-1
            topology[i,2] = Float64(_top_atomtype[7]).*1e3/Clapeyron.R̄
            Mw[i] = Float64(_top_atomtype[3])
        end
    else
        error("File type not supported")
    end

    # Rescale the coordinates to have the center of mass at 0,0,0

    com = sum(position.*Mw, dims=1)./sum(Mw)
    position .-= com
    position = [Tuple(position[i,:].*1e-9) for i in 1:natoms]

    return LJField(atoms, position, topology)
end

function evaluate_external_field(structure::DFTStructure,external_field::LJFieldModel,model::SAFTModel,Z)
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
        x = (zs[s][1].-Z[:,:,:,1]).^2
        y = (zs[s][2].-Z[:,:,:,2]).^2
        z = (zs[s][3].-Z[:,:,:,3]).^2
        r = sqrt.(x .+ y .+ z)
        ϵsi = sqrt.(ϵs[s].*ϵi)
        σsi = (σs[s].+σi)/2
        for i in 1:nbeads
            selectdim(external_field_values,nd+1,i) .+= @. 4*ϵsi[i]*((σsi[i]/r)^12-(σsi[i]/r)^6)
        end
    end
    return external_field_values./T
end

function evaluate_external_field(structure::DFTStructure,external_field::LJFieldModel,model::SAFTModel,ρ::Array{Float64},z)
    return evaluate_external_field(structure,external_field,model,z)
end