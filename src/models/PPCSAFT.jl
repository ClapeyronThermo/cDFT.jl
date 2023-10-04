function F_res(model::PPCSAFTModel,ρ,T,z,x)
    nc = length(model)
    idx = 1:nc
    # The below should be modified
    f(x) = f_mp(model,T,@view(x[idx]))
    Φ_polar = mapslices(f,ρ;dims=2)
    
    # return F_res(model::PCSAFTModel,ρ,T,z) + ∫(Φ_polar,dz)
end

function δFδρ_mp(model::PPCSAFTModel,ρ,T,z)
    # The below wouldn't work
    # return δFδρ_res(model::PCSAFTModel,ρ,T,z)+
           δFδρ_mp(model,ρ,T,z)
end

function f_mp(model::PPCSAFTModel, T, ρ̄, x)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    σ = model.params.sigma.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    η = π/6*sum(ρ̄.*m.*HSd.^3)

    # Gross and Vrabec, 2006, AIChE, 10.1002/aic.10683

    A_DD = A_DD(model,T,ρ̄,η,x)

    ã = 0

    return ã
end

function A_DD(model::PPCSAFTModel,T,ρ̄,η,x)
    x_norm = x ./ sum(x) # NEEDS REVISION necessary?
    
    m = model.params.segment.values
    ϵ = [model.params.epsilon.values[i,i] for i in @comps]
    σ = model.params.sigma.values
    μ̄² = model.params.dipole2.values

    μ⭒² = [μ̄²[i]/(m[i]*ϵ[i]*σ[i]^3) for i in @comps]

    # A₂
    A₂ = 0
    for i in @comps
        for j in @comps
            if i>j continue end
            A₂ += x_norm[i]*x_norm[j]*ϵ[i]*ϵ[j]/(k_B*T)* # ϵ_ii = ϵ_i
                (σ[i,i]*σ[j,j]/σ[i,j])^3*μ⭒²[i]*μ⭒²[j] # *n_1,n_2 are equal to unity, hence omitted
                J_DD_2ij(m[i],m[j],ϵ[i],ϵ[j],η,T)
        end
    end
    A₂ = -π*ρ̄ *A₂
    
    # A₃
    A₃ = 0
    for i in @comps
        for j in @comps
            if i>j continue end
            for k in @comps
                if j>k continue end
                σ_ij = (σ[i]+σ[j])/2
                σ_ik = (σ[i]+σ[k])/2
                σ_jk = (σ[j]+σ[k])/2
                A₃ += x_norm[i]*x_norm[j]*x_norm[k]*
                    ϵ[i]*ϵ[j]*ϵ[k]/(k_B*T)^2* # ϵ_ii = ϵ_i, σ_ii = σ_i
                    (σ[i]*σ[j]*σ[k])^3/(σ_ij*σ_ik*σ_jk)*μ⭒²[i]*μ⭒²[j]*μ⭒²[k]*
                    J_DD_3ijk(m[i],m[j],m[k],η)
            end
        end
    end
    A₃ = -4*(π^2)/3*A₃

    return A₂/(1-(A₃/A₂))
end

function J_DD_2ij(mᵢ,mⱼ,ϵᵢ,ϵⱼ,η,T)
    ϵ_ij = minimum([sqrt(ϵᵢ*ϵⱼ), 2.0]) # NEEDS REVISION minimum 2.0 needed?
    m_ij = minimum([sqrt(mᵢ*mⱼ), 2.0])
    corr_a = PPCSAFTconsts[:corr_a]
    corr_b = PPCSAFTconsts[:corr_b]
    m1 = (m_ij-1)/m_ij
    m2 = (m_ij-1)/m_ij*(m_ij-2)/m_ij

    J_DD_2ij = 0
    for n in 1:4
        a0n, a1n, a2n = corr_a[n]
        b0n, b1n, b2n = corr_b[n]
        a_nij = a0n + a1n*m1 + a2n*m2
        b_nij = b0n + b1n*m1 + b2n*m2
        J_DD_2ij += (a_nij+b_nij*ϵ_ij/(k_B*T))*η^n
    end

    return J_DD_2ij
end

function J_DD_3ijk(mᵢ,mⱼ,mₖ,η)
    m_ijk = minimum([cbrt(mᵢ*mⱼ*mₖ), 2.0])
    corr_c = PPCSAFTconsts[:corr_c]
    m1 = (m_ijk-1)/m_ijk
    m2 = (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk

    J_DD_3ijk = 0
    for n in 1:4
        c0n, c1n, c2n = corr_c[n]
        c_nijk = c0n + c1n*m1 + c2n*m2
        J_DD_3ijk += c_nijk*η^n
    end
    return J_DD_3ijk
end

function δFδρ_mp(model::PCSAFTModel,ρ,T,z)
    # See Sauer, Eqn 58
    return 0
end

const PPCSAFTconsts = (
    corr_a =
    ((0.3043504,0.9534641,-1.161008),
    (-0.1358588,-1.8396383,4.5258607),
    (1.4493329,2.013118,0.9751222),
    (0.3556977,-7.3724958,-12.281038),
    (-2.0653308,8.2374135,5.9397575)),

    corr_b =
    ((0.2187939,-0.5873164,3.4869576),
    (-1.1896431,1.2489132,-14.915974),
    (1.1626889,-0.508528,15.372022),
    (0.,0.,0.),
    (0.,0.,0.)),

    corr_c =
    ((-0.0646774,-0.9520876,-0.6260979),
    (0.1975882,2.9924258,1.2924686),
    (-0.8087562,-2.3802636,1.6542783),
    (0.6902849,-0.2701261,-3.4396744),
    (0.,0.,0.))
)