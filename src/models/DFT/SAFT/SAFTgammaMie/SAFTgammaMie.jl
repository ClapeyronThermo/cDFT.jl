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
