using Clapeyron: SAFTgammaMieModel
using Clapeyron: d_gc_av

function DFTSystem(model::SAFTgammaMieModel,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, nothing, propagator, options, chunksize)
end

function DFTSystem(model::SAFTgammaMieModel,structure::DFTStructure, external_field,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, external_field, propagator, options, chunksize)
end


struct SAFTgammaMieSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::SAFTgammaMie,structure::DFTStructure)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk 
    HSd = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk./sum(ρbulk)) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Bool.(model.groups.n_intergroups[i])
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = i_groups[findfirst(nbonds[i_groups] .== maximum(nbonds[i_groups]))]
        levels[i_root] = 1
    
        idx_current_level = i_root
        is_bonded = bond_mat[idx_current_level,:]
        k = 1
        while any(levels[i_groups] .== 0)
            levels[is_bonded] .= k+1
            idx_next_level = findall(levels .== k+1 .&& .!(is_leaf))
            is_bonded = (sum(bond_mat[idx_next_level,:],dims=1)[:].==1 .&& levels.==0)
            k+=1
        end
    end
    return SAFTgammaMieSpecies(nbeads,HSd,levels,ρbulk,μres)
end

function get_fields(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    nd = dimension(structure)
    ω = structure_ω(structure, device)
    d = species.size
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = d ./ σ
    ψ = @. cbrt(3*C*(1/(λ_a-3)-1/(λ_r-3)))
    return [SWeightedDensity(:ρ,zeros(nb),ω,ngrid,device),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device)]
end

function get_propagator(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device)
    return TangentHSPropagator(model, species, structure, device)
end



function expand_model(model::SAFTgammaMieModel) 
    
    nspecies = length(model)

    #Expand groups
    grouparam,ngroups_k = expand_groups(model)
    
    #Expand the sites 
    siteparams = expand_sites(model, grouparam, ngroups_k)
    params_old,vrparams_old = model.params,model.vrmodel.params
    PARAM = typeof(params_old)
    
    oldparams = PARAM(params_old.segment,params_old.shapefactor,params_old.lambda_a,params_old.lambda_r,params_old.sigma,params_old.epsilon,vrparams_old.epsilon_assoc,vrparams_old.bondvol,params_old.mixed_segment)
    #Expand the parameters
    eosparams = expand_params(oldparams, grouparam, siteparams, ngroups_k)

    #compute mixed segment
    Clapeyron.mix_segment!(eosparams.mixed_segment,grouparam,eosparams.shapefactor.values,eosparams.segment.values).values
    vreosparam_type = typeof(model.vrmodel.params)
    vreosparams = vreosparam_type(model.vrmodel.params.Mw,
                                  model.vrmodel.params.segment,
                                  model.vrmodel.params.sigma,
                                  model.vrmodel.params.lambda_a,
                                  model.vrmodel.params.lambda_r,
                                  model.vrmodel.params.epsilon,
                                  eosparams.epsilon_assoc,
                                  eosparams.bondvol)

    vrmodel = SAFTVRMie(model.vrmodel.components,
                        siteparams,
                        vreosparams,
                        model.vrmodel.idealmodel,
                        model.vrmodel.assoc_options,
                        model.vrmodel.references
                        )

    return new_model = SAFTgammaMie(model.components,
                                grouparam,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                vrmodel,
                                model.epsilon_mixing,
                                model.assoc_options,
                                model.references)
end

function f_res(system::DFTSystem, model::SAFTgammaMieModel, n)
    nd = dimension(system)
    n1,n2,n3,n4,n5,n6,n7 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4:4+nd-1,:]),@view(n[4+nd,:]),@view(n[5+nd,:]),@view(n[6+nd,:])

    return f_hs(system,model,n2,n3,n4) + f_disp(system,model,n7) + f_chain(system,model,n1,n5,n6) + f_assoc(system,model,n2,n3,n4)
end

function f_hs(system::DFTSystem, model::SAFTgammaMieModel, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values
    S = model.params.shapefactor.values
    HSd = species.size

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(nᵥ[:,1]), zero(nᵥ[:,1]), zero(n₀)
    for i in 1:length(n)
        mᵢ,Sᵢ,HSdᵢ,nᵥᵢ = m[i],S[i],HSd[i],nᵥ[:,i]
        nᵢmᵢ = n[i]*mᵢ*Sᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ .+= nᵥᵢ*mᵢ*Sᵢ/HSdᵢ
        nᵥ₂ .+= -2π*nᵥᵢ*mᵢ*Sᵢ
        n₃₃ += n₃[i]*mᵢ*Sᵢ
    end

    nᵥ₁nᵥ₂ = dot(nᵥ₁,nᵥ₂)
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)

    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₁nᵥ₂)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function f_disp(system::DFTSystem, model::SAFTgammaMieModel, ρ̄)
    V = nothing
    ψ = system.fields[end].width
    _d = system.species.size
    T = system.structure.conditions[2]
    m = model.params.segment.values
    S = model.params.shapefactor.values
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma

    m = m.*S

    ρ̄ = ρ̄*3 ./(4*ψ.^3)/π
    ∑ρ̄ = sum(ρ̄)
    z = ρ̄ /∑ρ̄
    m̄ = dot(z,m)
    m̄inv = 1/m̄
    ∑z = sum(z)

    ρS = dot(ρ̄,m)

    _ζ_X = zero(T+first(ρ̄)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = _ζ_X

    for i ∈ @groups
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        di =_d[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
            dij = (di + _d[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6
    
    a₁ = zero(T+first(z)+one(eltype(model)))
    a₂ = a₁
    a₃ = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS = @f(KHS,_ζ_X,ρS)
    for i ∈ @groups
        j = i
        x_Si = z[i]*m[i]*m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i,j]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        dij3 = dij^3
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a = @f(aS_1,λa,_ζ_X)
        aS_1_r = @f(aS_1,λr,_ζ_X)
        B_a = @f(B,λa,x_0ij,_ζ_X)
        B_r = @f(B,λr,x_0ij,_ζ_X)
        a1_ij = (2*π*ϵ*dij3)*_C*ρS*
        (x_0ij^λa*(aS_1_a+B_a) - x_0ij^λr*(aS_1_r+B_r))

        #calculations for a2 - diagonal
        aS_1_2a = @f(aS_1,2*λa,_ζ_X)
        aS_1_2r = @f(aS_1,2*λr,_ζ_X)
        aS_1_ar = @f(aS_1,λa+λr,_ζ_X)
        B_2a = @f(B,2*λa,x_0ij,_ζ_X)
        B_2r = @f(B,2*λr,x_0ij,_ζ_X)
        B_ar = @f(B,λr+λa,x_0ij,_ζ_X)
        α = _C*(1/(λa-3)-1/(λr-3))
        f1,f2,f3,f4,f5,f6 = @f(f123456,α)
        _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
        a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
        (x_0ij^(2*λa)*(aS_1_2a+B_2a)
        - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
        + x_0ij^(2*λr)*(aS_1_2r+B_2r))

        #calculations for a3 - diagonal
        a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
        #adding - diagonal
        a₁ += a1_ij*x_Si*x_Si
        a₂ += a2_ij*x_Si*x_Si
        a₃ += a3_ij*x_Si*x_Si
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            ϵ = _ϵ[i,j]
            λa = _λa[i,j]
            λr = _λr[i,j]
            σ = _σ[i,j]
            _C = @f(Cλ,λa,λr)
            dij = 0.5*(_d[i]+_d[j])
            x_0ij = σ/dij
            dij3 = dij^3
            x_0ij = σ/dij
            #calculations for a1
            a1_ij = (2*π*ϵ*dij3)*_C*ρS*
            (x_0ij^λa*(@f(aS_1,λa,_ζ_X)+@f(B,λa,x_0ij,_ζ_X)) - x_0ij^λr*(@f(aS_1,λr,_ζ_X)+@f(B,λr,x_0ij,_ζ_X)))

            #calculations for a2
            α = _C*(1/(λa-3)-1/(λr-3))
            f1,f2,f3,f4,f5,f6 = @f(f123456,α)
            _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
            a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
            (x_0ij^(2*λa)*(@f(aS_1,2*λa,_ζ_X)+@f(B,2*λa,x_0ij,_ζ_X))
            - 2*x_0ij^(λa+λr)*(@f(aS_1,λa+λr,_ζ_X)+@f(B,λa+λr,x_0ij,_ζ_X))
            + x_0ij^(2*λr)*(@f(aS_1,2λr,_ζ_X)+@f(B,2*λr,x_0ij,_ζ_X)))

            #calculations for a3
            a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
            #adding
            a₁ += 2*a1_ij*x_Si*x_Sj
            a₂ += 2*a2_ij*x_Si*x_Sj
            a₃ += 2*a3_ij*x_Si*x_Sj
        end
    end
    a₁ = a₁*m̄/T/∑z #/sum(z)
    a₂ = a₂*m̄/(T*T)/∑z  #/sum(z)
    a₃ = a₃*m̄/(T*T*T)/∑z  #/sum(z)
    #@show (a₁,a₂,a₃)
    adisp = a₁ + a₂ + a₃
    return ∑ρ̄*adisp
end

function f_chain(system::DFTSystem, model::SAFTgammaMieModel, ρhc, ρ̄hc, _λ)
    V = nothing
    T = system.structure.conditions[2]
    x = system.structure.ρbulk / sum(system.structure.ρbulk)
    
    m = model.vrmodel.params.segment
    m_gc = model.params.segment.values .* model.params.shapefactor.values
    _ϵ = model.vrmodel.params.epsilon
    _λr = model.vrmodel.params.lambda_r
    _λa = model.vrmodel.params.lambda_a
    _σ = model.vrmodel.params.sigma
    _σ_gc = model.params.sigma.values
    _d = d_gc_av(model,V,T,x,system.species.size)

    ρ̄hc = ρ̄hc*3 ./(4 .*system.species.size.^3)/π
    _λ =_λ ./ (2*system.species.size)

    _ρhc = zeros(eltype(ρhc),length(model))
    _ρ̄hc = zeros(eltype(ρ̄hc),length(model))
    λ = zeros(eltype(_λ),length(model))

    for i in @comps
        for k in @groups(i)
            _ρhc[i] += ρhc[k]/system.species.nbeads[i]
            _ρ̄hc[i] += ρ̄hc[k]/system.species.nbeads[i]
            λ[i] += _λ[k]/system.species.nbeads[i]
        end
    end

    z = _ρ̄hc /sum(_ρ̄hc)
    z_gc = ρhc /sum(ρhc)

    m̄ = dot(z,m)
    m̄_gc = dot(z_gc,m_gc)
    m̄inv_gc = 1/m̄_gc

    ρS = dot(_ρ̄hc,m)

    _ζ_X = zero(T+first(_ρ̄hc)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = deepcopy(_ζ_X)

    for i ∈ @groups
        x_Si = z_gc[i]*m_gc[i]*m̄inv_gc
        σ3_x += x_Si*x_Si*(_σ_gc[i,i]^3)
        di =system.species.size[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z_gc[j]*m_gc[j]*m̄inv_gc
            σ3_x += 2*x_Si*x_Sj*(_σ_gc[i,j]^3)
            dij = (di + system.species.size[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6

    fchain = zero(T+first(z)+one(eltype(model)))
    _KHS,_∂KHS = @f(KHS_fdf,_ζ_X,ρS)
    for i ∈ @comps
        ϵ = _ϵ[i,i]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        x_0ij = σ/dij
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a,∂aS_1∂ρS_a = @f(aS_1_fdf,λa,_ζ_X,ρS)
        aS_1_r,∂aS_1∂ρS_r = @f(aS_1_fdf,λr,_ζ_X,ρS)
        B_a,∂B∂ρS_a = @f(B_fdf,λa,x_0ij,_ζ_X,ρS)
        B_r,∂B∂ρS_r = @f(B_fdf,λr,x_0ij,_ζ_X,ρS)

        #calculations for a2 - diagonal
        aS_1_2a,∂aS_1∂ρS_2a = @f(aS_1_fdf,2*λa,_ζ_X,ρS)
        aS_1_2r,∂aS_1∂ρS_2r = @f(aS_1_fdf,2*λr,_ζ_X,ρS)
        aS_1_ar,∂aS_1∂ρS_ar = @f(aS_1_fdf,λa+λr,_ζ_X,ρS)
        B_2a,∂B∂ρS_2a = @f(B_fdf,2*λa,x_0ij,_ζ_X,ρS)
        B_2r,∂B∂ρS_2r = @f(B_fdf,2*λr,x_0ij,_ζ_X,ρS)
        B_ar,∂B∂ρS_ar = @f(B_fdf,λr+λa,x_0ij,_ζ_X,ρS)
        α = _C*(1/(λa-3)-1/(λr-3))
        g_HSi = @f(g_HS,x_0ij,_ζ_X)
        #@show (g_HSi,i)
        ∂a_1∂ρ_S = _C*(x_0ij^λa*(∂aS_1∂ρS_a+∂B∂ρS_a)
                      - x_0ij^λr*(∂aS_1∂ρS_r+∂B∂ρS_r))
        #@show (∂a_1∂ρ_S,1)

        g_1_ = 3*∂a_1∂ρ_S-_C*(λa*x_0ij^λa*(aS_1_a+B_a)-λr*x_0ij^λr*(aS_1_r+B_r))
        #@show (g_1_,i)
        θ = exp(ϵ/T)-1
        γc = 10 * (-tanh(10*(0.57-α))+1) * _ζst*θ*exp(-6.7*_ζst-8*_ζst^2)
        ∂a_2∂ρ_S = 0.5*_C^2 *
            (ρS*_∂KHS*(x_0ij^(2*λa)*(aS_1_2a+B_2a)
            - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
            + x_0ij^(2*λr)*(aS_1_2r+B_2r))
            + _KHS*(x_0ij^(2*λa)*(∂aS_1∂ρS_2a+∂B∂ρS_2a)
            - 2*x_0ij^(λa+λr)*(∂aS_1∂ρS_ar+∂B∂ρS_ar)
            + x_0ij^(2*λr)*(∂aS_1∂ρS_2r+∂B∂ρS_2r)))

        gMCA2 = 3*∂a_2∂ρ_S-_KHS*_C^2 *
        (λr*x_0ij^(2*λr)*(aS_1_2r+B_2r)-
            (λa+λr)*x_0ij^(λa+λr)*(aS_1_ar+B_ar)+
            λa*x_0ij^(2*λa)*(aS_1_2a+B_2a))
        g_2_ = (1+γc)*gMCA2
        #@show (g_2_,i)
        g_Mie_ = g_HSi*exp(ϵ/T*g_1_/g_HSi+(ϵ/T)^2*g_2_/g_HSi)
        #@show (g_Mie_,i)
        fchain +=  _ρhc[i]*(log(g_Mie_)*(m[i]-1))
    end
    
    return -fchain
end

function Δ(model::SAFTgammaMieModel, T, n, n₃, nᵥ, i, j, a, b)
    _d = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))
    _σ = model.params.sigma.values
    m = model.params.segment.values
    S = model.params.shapefactor.values
    ϵ_assoc = model.params.epsilon_assoc.values
    K = model.params.bondvol.values[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(K))
    iszero(K) && return _0

    ρ̄ = n₃*3*2 ./(_d.^3)/π
    m = m.*S
    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = dot(ρ̄,m)

    σ3_x = zero(T+first(z)+one(eltype(model)))

    for i ∈ @groups
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
        end
    end
    ρr  = ρS*σ3_x
    
    ϵ = model.vrmodel.params.epsilon
    Tr = T/ϵ[i,j]
    _I = I(model,Tr,ρr)
    
    F = expm1(ϵ_assoc[i,j][a,b]/T)

    return F*K*_I
end

function I(model::SAFTgammaMieModel, Tr,ρr)
    c  = SAFTVRMieconsts.c
    res = zero(ρr+Tr)
    @inbounds for n ∈ 0:10
        ρrn = ρr^n
        res_m = zero(res)
        for m ∈ 0:(10-n)
            res_m += c[n+1,m+1]*Tr^m
        end
        res += res_m*ρrn
    end
    return res
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for SAFTγMie: FMT hard-sphere (with shapefactor) +
chain (groups aggregated to species level, gMie from vrmodel params) +
SAFT-VR Mie dispersion (with effective m*S segments).

Field layout (same as SAFTVRMieModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (for TangentHSPropagator chain)
  5+ND     : ∫ρdz  with d     → λ    (for TangentHSPropagator chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)

NC here is the total number of groups (sum of nbeads per component).
"""
@inline function f_chain(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    _pi   = 3.141592653589793
    eps_v = 1e-15

    HSd    = params.HSd
    meff   = params.meff
    σ      = params.sigma
    A      = params.A
    ϕ      = params.phi

    idx_ζ_c = 4 + ND
    nbeads_c = params.nbeads_comp
    HSd_s    = params.HSd_species
    m_s      = params.m_species
    σ_s      = params.sigma_species
    ϵ_s      = params.epsilon_species
    λr_s     = params.lambda_r_species
    λa_s     = params.lambda_a_species
    nc_s     = length(nbeads_c)

    ρS_c = eps_v
    g_off = 1
    @inbounds for s in 1:nc_s
        nb_s = nbeads_c[s]
        ρ̄hc_s = 0.0
        @inbounds for kg in g_off:(g_off + nb_s - 1)
            dg = HSd[kg]
            ρ̄hc_s += n[kk, idx_ζ_c, kg] * 3.0/(4.0*_pi*dg^3)
        end
        ρ̄hc_s /= Float64(nb_s)
        ρS_c += ρ̄hc_s * m_s[s]
        g_off += nb_s
    end
    kρS_c = ρS_c * _pi/6.0/8.0

    ρhc_gc_total = eps_v
    @inbounds for kg in 1:NC
        ρhc_gc_total += n[kk, 1, kg]
    end
    m̄_gc = 0.0
    @inbounds for kg in 1:NC
        z_gc_kg = n[kk, 1, kg] / ρhc_gc_total
        m̄_gc += z_gc_kg * meff[kg]
    end
    m̄inv_gc = 1.0/(m̄_gc + eps_v)

    ζ_Xc = 0.0;  σ3_xc = 0.0
    @inbounds for i in 1:NC
        z_gc_i = n[kk, 1, i] / ρhc_gc_total
        x_Si_c = z_gc_i * meff[i] * m̄inv_gc
        di_c   = HSd[i]
        σ3_xc += x_Si_c*x_Si_c*σ[i,i]^3
        ζ_Xc  += kρS_c*x_Si_c*x_Si_c*(2.0*di_c)^3
        @inbounds for j in 1:(i-1)
            z_gc_j = n[kk, 1, j] / ρhc_gc_total
            x_Sj_c = z_gc_j * meff[j] * m̄inv_gc
            dj_c   = HSd[j]
            σ3_xc += 2.0*x_Si_c*x_Sj_c*σ[i,j]^3
            ζ_Xc  += 2.0*kρS_c*x_Si_c*x_Sj_c*(di_c+dj_c)^3
        end
    end
    ζstc = σ3_xc * ρS_c * _pi/6.0
    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = 0.0
    g_off = 1
    @inbounds for s in 1:nc_s
        nb_s = nbeads_c[s]
        ρhc_s = 0.0
        @inbounds for kg in g_off:(g_off + nb_s - 1)
            ρhc_s += n[kk, 1, kg]
        end
        ρhc_s /= Float64(nb_s)

        di_s = HSd_s[s]
        λa_c = λa_s[s,s];  λr_c = λr_s[s,s]
        _Cc  = _Cλ_kernel(λa_c, λr_c)
        x0c  = σ_s[s,s] / di_s
        ϵiic = ϵ_s[s,s]

        aS1c_a,  dS1c_a  = _aS1_fdf_kernel(λa_c,       ζ_Xc, A)
        aS1c_r,  dS1c_r  = _aS1_fdf_kernel(λr_c,       ζ_Xc, A)
        Bc_a,    dBc_a   = _B_fdf_kernel(λa_c,     x0c, ζ_Xc)
        Bc_r,    dBc_r   = _B_fdf_kernel(λr_c,     x0c, ζ_Xc)
        aS1c_2a, dS1c_2a = _aS1_fdf_kernel(2.0*λa_c,   ζ_Xc, A)
        aS1c_2r, dS1c_2r = _aS1_fdf_kernel(2.0*λr_c,   ζ_Xc, A)
        aS1c_ar, dS1c_ar = _aS1_fdf_kernel(λa_c+λr_c,  ζ_Xc, A)
        Bc_2a,   dBc_2a  = _B_fdf_kernel(2.0*λa_c, x0c, ζ_Xc)
        Bc_2r,   dBc_2r  = _B_fdf_kernel(2.0*λr_c, x0c, ζ_Xc)
        Bc_ar,   dBc_ar  = _B_fdf_kernel(λa_c+λr_c,x0c, ζ_Xc)

        ∂a1ρSc = _Cc*(x0c^λa_c*(dS1c_a+dBc_a) - x0c^λr_c*(dS1c_r+dBc_r))
        g1c    = 3.0*∂a1ρSc - _Cc*(λa_c*x0c^λa_c*(aS1c_a+Bc_a) - λr_c*x0c^λr_c*(aS1c_r+Bc_r))

        αc  = _Cc*(1.0/(λa_c-3.0) - 1.0/(λr_c-3.0))
        f1c,f2c,f3c,f4c,f5c,f6c = _f123456_kernel(αc, ϕ)
        θc  = exp(ϵiic/T) - 1.0
        γcc = 10.0*(-tanh(10.0*(0.57-αc))+1.0)*ζstc*θc*exp(-6.7*ζstc-8.0*ζstc^2)

        cb2ac = x0c^(2.0*λa_c)*(aS1c_2a+Bc_2a)
        cbarc = x0c^(λa_c+λr_c)*(aS1c_ar+Bc_ar)
        cb2rc = x0c^(2.0*λr_c)*(aS1c_2r+Bc_2r)
        ∂a2ρSc = 0.5*_Cc*_Cc*(
            ρS_c*_∂KHSc*(cb2ac - 2.0*cbarc + cb2rc)
          + _KHSc*(x0c^(2.0*λa_c)*(dS1c_2a+dBc_2a)
                 - 2.0*x0c^(λa_c+λr_c)*(dS1c_ar+dBc_ar)
                 + x0c^(2.0*λr_c)*(dS1c_2r+dBc_2r))
        )
        gMCA2c = 3.0*∂a2ρSc - _KHSc*_Cc*_Cc*(λr_c*cb2rc - (λa_c+λr_c)*cbarc + λa_c*cb2ac)
        g2c    = (1.0+γcc)*gMCA2c

        gHSc  = _gHS_kernel(x0c, ζ_Xc)
        gMiec = gHSc * exp(ϵiic/T * g1c/gHSc + (ϵiic/T)^2 * g2c/gHSc)

        ms = m_s[s]
        res_chain += ρhc_s * Base.log(abs(gMiec) + eps_v) * (ms - 1.0)

        g_off += nb_s
    end
    return -res_chain
end

@inline function f_res(out, n, params, T, kk,
                       ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    res_hs, = f_hs(n, params.meff, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_disp = f_disp_mie(n, params.meff, params.HSd, params.sigma, params.epsilon,
                           params.lambda_r, params.lambda_a, params.psi_eff,
                           kk, T, Val(NC), Val(ND), Val(6+ND), params.A, params.phi)
    res_chain = f_chain(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_chain + res_disp
    return nothing
end

function preallocate_params(system::DFTSystem{<:SAFTgammaMieModel})
    backend = system.options.device
    T_val  = system.structure.conditions[2]
    x_val  = system.structure.ρbulk ./ sum(system.structure.ρbulk)
    HSd_sp = d_gc_av(system.model, 1e-3, T_val, x_val, system.species.size)

    m_vals = system.model.params.segment.values
    S_vals = system.model.params.shapefactor.values
    meff   = m_vals .* S_vals

    params = (;
        HSd               = Adapt.adapt(backend, system.species.size),
        m                 = Adapt.adapt(backend, m_vals),
        S                 = Adapt.adapt(backend, S_vals),
        meff              = Adapt.adapt(backend, meff),
        sigma             = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon           = Adapt.adapt(backend, system.model.params.epsilon.values),
        lambda_r          = Adapt.adapt(backend, system.model.params.lambda_r.values),
        lambda_a          = Adapt.adapt(backend, system.model.params.lambda_a.values),
        psi_eff           = Adapt.adapt(backend, system.fields[end].width),
        A                 = SAFTVRMIE_A,
        phi               = SAFTVRMIE_PHI,
        nbeads_comp       = Adapt.adapt(backend, system.species.nbeads),
        HSd_species       = Adapt.adapt(backend, HSd_sp),
        m_species         = Adapt.adapt(backend, system.model.vrmodel.params.segment.values),
        sigma_species     = Adapt.adapt(backend, system.model.vrmodel.params.sigma.values),
        epsilon_species   = Adapt.adapt(backend, system.model.vrmodel.params.epsilon.values),
        lambda_r_species  = Adapt.adapt(backend, system.model.vrmodel.params.lambda_r.values),
        lambda_a_species  = Adapt.adapt(backend, system.model.vrmodel.params.lambda_a.values),
    )
    nc = sum(system.species.nbeads)
    return params, nc
end
