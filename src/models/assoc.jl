function f_assoc(system::DFTSystem, model::SAFTModel, n, n₃, nᵥ)
    HSd = system.species.size
    (_, T, _) = system.structure.conditions
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ))
    nn = assoc_pair_length(model)
    iszero(nn) && return _0

    n₀ = n./HSd
    n₂ = π.*HSd.*n
    nᵥ₂ = -2π.*nᵥ
    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2
    isone(nn) && return f_assoc_exact_1(model, T, n, n₃, nᵥ, n₀, ξ)

    X_ = X(model,T,n,n₃,nᵥ,n₀,ξ)
    _0 = zero(first(X_.v))

    ns = model.sites.n_sites
    res = _0
    resᵢₐ = _0
    for i ∈ @comps
        ni = ns[i]
        iszero(length(ni)) && continue
        Xᵢ = X_[i]
        resᵢₐ = _0
        for (a,nᵢₐ) ∈ pairs(ni)
            Xᵢₐ = Xᵢ[a]
            nᵢₐ = ni[a]
            resᵢₐ +=  nᵢₐ* (log(Xᵢₐ) - Xᵢₐ/2 + 0.5)
        end
        res += resᵢₐ*n₀[i]*ξ[i]
    end
    return res
end

function Δ(model::EoSModel, T, n, n₃, nᵥ)
    Δout = assoc_similar(model,typeof(T+first(n₃)+first(n)+first(nᵥ)))
    Δout.values .= false
    for (idx,(i,j),(a,b)) in indices(Δout)
        Δout[idx] =Δ(model,T,n,n₃,nᵥ,i,j,a,b)
    end
    return Δout
end

function X(system::DFTSystem, model::EoSModel, n, n₃, nᵥ)
    HSd = system.species.size
    n₀ = n./HSd
    n₂ = π.*HSd.*n
    nᵥ₂ = -2π.*nᵥ
    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2
    return X(model,T,n,n₃,nᵥ,n₀,ξ)
end

function X(model::EoSModel,T,n,n₃,nᵥ,n₀,ξ)
    options = assoc_options(model)
    nn = assoc_pair_length(model)
    isone(nn) && return X_exact1(model,T,n,n₃,nᵥ,n₀,ξ)
    K = assoc_site_matrix(model,T,n,n₃,nᵥ,n₀,ξ)
    idxs = model.sites.n_sites.p
    Xsol = assoc_matrix_solve(K,options)
    return PackedVofV(idxs,Xsol)
end

function assoc_site_matrix(system::DFTSystem, model::EoSModel, n, n₃, nᵥ)
    HSd = system.species.size
    n₀ = n./HSd
    n₂ = π.*HSd.*n
    nᵥ₂ = -2π.*nᵥ
    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2
    return assoc_site_matrix(model,T,n,n₃,nᵥ,n₀,ξ)
end

function assoc_site_matrix(model::EoSModel,T,n,n₃,nᵥ,n₀,ξ)
    delta = Δ(model,T,n,n₃,nᵥ)
    sitesparam = Clapeyron.getsites(model)
    _sites = sitesparam.n_sites
    p = _sites.p
    _ii::Vector{Tuple{Int,Int}} = delta.outer_indices
    _aa::Vector{Tuple{Int,Int}} = delta.inner_indices
    _idx = 1:length(_ii)
    _Δ= delta.values
    TT = eltype(_Δ)
    count = 0
    _n = sitesparam.n_sites.v
    nn = length(_n)
    K  = zeros(TT,nn,nn)
    count = 0
    options = assoc_options(model)
    combining = options.combining
    @inbounds for i ∈ 1:length(model) #for i ∈ comps
        sitesᵢ = 1:(p[i+1] - p[i]) #sites are normalized, with independent indices for each component
        for a ∈ sitesᵢ #for a ∈ sites(comps(i))
            ia = compute_index(p,i,a)
            for idx ∈ _idx #iterating for all sites
                ij = _ii[idx]
                ab = _aa[idx]
                if issite(i,a,ij,ab)
                    j = complement_index(i,ij)
                    b = complement_index(a,ab)
                    jb = compute_index(p,j,b)
                    njb = _n[jb]
                    K[ia,jb]  = n₀[j]*ξ[j]*njb*_Δ[idx]
                end
            end
        end
    end
    return K
end

function f_assoc_exact_1(model, V, T, n, n₃, nᵥ, n₀, ξ)
    xia,xjb,i,j,a,b,_n,idxs = _X_exact1(model, V, T, n, n₃, nᵥ, n₀, ξ)
    _0 = zero(xia)
    sites = getsites(model)
    nn = sites.n_sites
    res = _0
    resᵢₐ = _0
    nia = nn[i][a]
    njb = nn[j][b]
    n₀ᵢ,n₀ⱼ = n₀[i], n₀[j]
    ξᵢ,ξⱼ = ξ[i], ξ[j]
    res = ξᵢ*n₀ᵢ*nia*(log(xia) - xia*0.5 + 0.5)
    if (i != j) | (a != b) #we check if we have 2 sites or just 1
        res += n₀ⱼ*ξⱼ*njb*(log(xjb) - xjb*0.5 + 0.5)
    end
    return res
end

function X_exact1(model, T, n, n₃, nᵥ, n₀, ξ)
    xia,xjb,i,j,a,b,n,idxs = _X_exact1(model, T, n, n₃, nᵥ, n₀, ξ)
    Clapeyron.pack_X_exact1(xia,xjb,i,j,a,b,n,idxs)
end

function _X_exact1(model, T, n, n₃, nᵥ, n₀, ξ)
    κ = model.params.bondvol.values
    i,j = κ.outer_indices[1]
    a,b = κ.inner_indices[1]
    _Δ = Δ(model, T, n, n₃, nᵥ, i, j, a, b)
    _1 = one(eltype(_Δ))
    sitesparam = getsites(model)
    idxs = sitesparam.n_sites.p
    _n = length(sitesparam.n_sites.v)
    ni = sitesparam.n_sites[i]
    na = ni[a]
    nj = sitesparam.n_sites[j]
    nb = nj[b]
    n₀ᵢ,n₀ⱼ = n₀[i], n₀[j]
    ξᵢ,ξⱼ = ξ[i], ξ[j]
    kia = na*n₀ᵢ*ξᵢ*_Δ
    kjb = nb*n₀ⱼ*ξⱼ*_Δ
    _a = kia
    _b = _1 - kia + kjb
    _c = -_1
    denom = _b + sqrt(_b*_b - 4*_a*_c)
    xia = -2*_c/denom
    xk_ia = kia*xia
    xjb = (1- xk_ia)/(1 - xk_ia*xk_ia)
    return xia,xjb,i,j,a,b,_n,idxs
end