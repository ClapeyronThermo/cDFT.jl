function Δ(model::EoSModel, T, n, n₃, nᵥ)
    Δout = assoc_similar(model,typeof(T+first(n₃)+first(n)+first(nᵥ)))
    Δout.values .= false
    for (idx,(i,j),(a,b)) in indices(Δout)
        Δout[idx] =Δ(model,T,n, n₃, nᵥ,i,j,a,b)
    end
    return Δout
end

function X(model::EoSModel,T,n,n₃,nᵥ)
    options = assoc_options(model)
    K = assoc_site_matrix(model,T,n,n₃,nᵥ)
    idxs = model.sites.n_sites.p
    Xsol = assoc_matrix_solve(K,options)
    return PackedVofV(idxs,Xsol)
end

function assoc_site_matrix(model::EoSModel,T,n,n₃,nᵥ)
    HSd = d(model,1e-3,T,onevec(model))

    n₀ = n./HSd
    n₂ = π.*HSd.*n

    nᵥ₂ = -2π.*nᵥ

    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2

    delta = Δ(model,T,n,n₃,nᵥ)
    _sites = model.sites.n_sites
    p = _sites.p
    _ii::Vector{Tuple{Int,Int}} = delta.outer_indices
    _aa::Vector{Tuple{Int,Int}} = delta.inner_indices
    _idx = 1:length(_ii)
    _Δ= delta.values
    TT = eltype(_Δ)
    count = 0
    @inbounds for i ∈ 1:length(n) #for i ∈ comps
        sitesᵢ = 1:(p[i+1] - p[i]) #sites are normalized, with independent indices for each component
        for a ∈ sitesᵢ #for a ∈ sites(comps(i))
            #ia = compute_index(pack_indices,i,a)
            for idx ∈ _idx #iterating for all sites
                ij = _ii[idx]
                ab = _aa[idx]
                issite(i,a,ij,ab) && (count += 1)
            end
        end
    end
    c1 = zeros(Int,count)
    c2 = zeros(Int,count)
    val = zeros(TT,count)
    _n = model.sites.n_sites.v
    count = 0
    @inbounds for i ∈ 1:length(n) #for i ∈ comps
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
                    count += 1
                    c1[count] = ia
                    c2[count] = jb
                    val[count] = n₀[j]*ξ[j]*njb*_Δ[idx]
                end
            end
        end
    end
    K::SparseMatrixCSC{TT,Int} = sparse(c1,c2,val)
    return K
end