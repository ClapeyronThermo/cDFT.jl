function F_res(model::PPCSAFTModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    dz = ρ[1].mesh_size
    
    _, ρ̄, _    = weights_hs(model,ρ,z,ψ*HSd)

    f(x) = f_mp(model,T,@view(x[@comps]))
    Φ_mp = mapslices(f,ρ̄;dims=2)

    _F_res_PCSAFT = invoke(F_res, Tuple{PCSAFTModel,Any,Any,Any},model,ρ,T,z)
    return _F_res_PCSAFT + ∫(Φ_mp,dz)
end

function δFδρ_res(model::PPCSAFTModel,ρ,T,z)
    _δFδρ_res_PCSAFT = invoke(δFδρ_res, Tuple{PCSAFTModel,Any,Any,Any},model,ρ,T,z)
    return _δFδρ_res_PCSAFT + δFδρ_mp(model,ρ,T,z)
end

function δFδρ_mp(model::PPCSAFTModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    lim = ψ*HSd

    _, ρ̄, _  = weights_hs(model,ρ,z,lim)

    f(x) = f_mp(model,T,@view(x[@comps]))
    df(x) = ForwardDiff.gradient(f,x)
    
    δfδn0  = mapslices(df,ρ̄;dims=2)
    ∂f∂n0 = δfδn0[:,@comps]

    δFδρ_mp = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+(-lim[i],lim[i])
        ∂f∂n =  DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
    
        span = range(-lim[i],lim[i],length=101) # Length = 101? Is it because len(z) = 101?

        δFδρ_mp[:,i] = π*∫ρz²dz.(Ref(∂f∂n),z,Ref(span))
    end

    return δFδρ_mp
end

function f_mp(model::PPCSAFTModel,T,ρ̄)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    η = π/6*sum(ρ̄.*m.*HSd.^3)
    x = ρ̄ /sum(ρ̄)

    # Gross and Vrabec, 2006, AIChE, 10.1002/aic.10683

    _f_DD = f_DD(model,T,ρ̄,η,x)
    # Leaving for expandability

    return _f_DD
end

# Dipole-Dipole Interaction
function f_DD(model::PPCSAFTModel,T,ρ̄,η,x)
    x_norm = x ./ sum(x)
    
    m = model.params.segment.values
    ϵ = [model.params.epsilon.values[i,i] for i in @comps]
    σ = model.params.sigma.values
    μ̄² = model.params.dipole2.values

    ρ̄ = sum(ρ̄)

    # TODO: Optimize the code below: Too many same calculations
    A₂ = 0
    for i in @comps
        for j in @comps
            A₂ += x_norm[i]*x_norm[j]*
                1/T^2*1/σ[i,j]^3*μ̄²[i]*μ̄²[j]*                
                J_DD_2ij(m[i],m[j],ϵ[i],ϵ[j],η,T)
        end
    end
    A₂ *= -π*ρ̄

    # A₃
    A₃ = 0
    for i in @comps
        for j in @comps
            for k in @comps
                A₃ += x_norm[i]*x_norm[j]*x_norm[k]*
                    1/T^3*1/(σ[i,j]*σ[j,k]*σ[i,k])*μ̄²[i]*μ̄²[j]*μ̄²[k]*
                    J_DD_3ijk(m[i],m[j],m[k],η)
            end
        end
    end
    A₃ *= -4*(π^2)/3*ρ̄^2
    

    A_DD = A₂^2/(A₂-A₃)

    return A_DD*ρ̄
end

function J_DD_2ij(mᵢ,mⱼ,ϵᵢ,ϵⱼ,η,T)
    ϵ_ij = sqrt(ϵᵢ*ϵⱼ) # NEEDS REVISION minimum 2.0 needed?
    m_ij = minimum([sqrt(mᵢ*mⱼ), 2.0])
    corr_a = PPCSAFTconsts[:corr_a]
    corr_b = PPCSAFTconsts[:corr_b]
    m1 = (m_ij-1)/m_ij
    m2 = (m_ij-1)/m_ij*(m_ij-2)/m_ij

    J_DD_2ij = 0
    for n in 0:4
        a0n, a1n, a2n = corr_a[n+1]
        b0n, b1n, b2n = corr_b[n+1]
        a_nij = a0n + a1n*m1 + a2n*m2
        b_nij = b0n + b1n*m1 + b2n*m2
        J_DD_2ij += (a_nij+b_nij*ϵ_ij/T)*η^n
    end

    return J_DD_2ij
end

function J_DD_3ijk(mᵢ,mⱼ,mₖ,η)
    m_ijk = minimum([cbrt(mᵢ*mⱼ*mₖ), 2.0])
    corr_c = PPCSAFTconsts[:corr_c]
    m1 = (m_ijk-1)/m_ijk
    m2 = (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk

    J_DD_3ijk = 0
    for n in 0:4
        c0n, c1n, c2n = corr_c[n+1]
        c_nijk = c0n + c1n*m1 + c2n*m2
        J_DD_3ijk += c_nijk*η^n
    end
    return J_DD_3ijk
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