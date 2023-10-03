function F_res(model::PPCSAFTModel,ρ,T,z)
    nc = length(model)
    idx = 1:nc
    # The below should be modified
    f(x) = f_mp(model,T,@view(x[idx]))
    Φ_polar = mapslices(f,ρ;dims=2)
    
    return F_res(model::PCSAFTModel,ρ,T,z) + ∫(Φ_polar,dz)
end

function δFδρ_res(model::PPCSAFTModel,ρ,T,z)
    return δFδρ_res(model::PCSAFTModel,ρ,T,z)+
           δFδρ_mp(model,ρ,T,z)
end

function f_mp(model::PPCSAFTModel, T, ρ̄)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    σ = model.params.sigma.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π

    # Gross and Vrabec, 2006, AIChE, 10.1002/aic.10683

    A₂ = A₂(ρ̄,z)
    A₃ = A₃(ρ̄,z)
    A_DD = A₂/(1-(A₃/A₂))

    ã = 0

    return ã
end

function A₂(ρ,z)
    a = 1
    return -π*a
end

function A₃(ρ,z)
    a = 1
    return -4*(π^2)/3*a
end

path = "../../database/PPCSAFT/model_constants_dipole_contribution.csv"
coef_dp = CSV.File(path) |> DataFrame


function J_DD_2ij(mᵢ,mⱼ,ϵᵢ,ϵⱼ,ρ,T)
    ϵ_ij = minimum([sqrt(ϵᵢ*ϵⱼ), 2.0]) # NEEDS REVISION
    m_ij = minimum([sqrt(mᵢ*mⱼ), 2.0])
    η = 0 # dimensionless density, related to ρ, TODO
    J_DD_2ij = 0
    for n in 1:4
        a_nij = _corr("a",n,m_ij)
        b_nij = _corr("b",n,m_ij)
        J_DD_2ij += (a_nij+b_nij*ϵ_ij/(k_B*T))*η^n
    end
    return J_DD_2ij
end

function J_DD_3ijk(mᵢ,mⱼ,mₖ,ρ)
    m_ijk = minimum([cbrt(mᵢ*mⱼ*mₖ), 2.0])
    η = 0 # dimensionless density, related to ρ, TODO
    J_DD_3ijk = 0
    for n in 1:4
        c_nijk = _corr("c",n,m_ijk)
        J_DD_3ijk += c_nijk*η^n
    end
    return J_DD_3ijk
end

function _corr(a,n,m)
    a0n, a1n, a2n = PPCSAFTconsts["corr_$a"][n]
    return a0n + a1n*(m-1)/m + a2n*(m-1)/m*(m-2)/m
end

function δFδρ_mp(model::PCSAFTModel,ρ,T,z)
    # See Sauer, Eqn 58
    return 0
end

const PPCSAFTconsts = Dict(
    "corr_a" =>
    ((0.3043504,0.9534641,-1.161008),
    (-0.1358588,-1.8396383,4.5258607),
    (1.4493329,2.013118,0.9751222),
    (0.3556977,-7.3724958,-12.281038),
    (-2.0653308,8.2374135,5.9397575)),

    "corr_b" =>
    ((0.2187939,-0.5873164,3.4869576),
    (-1.1896431,1.2489132,-14.915974),
    (1.1626889,-0.508528,15.372022),
    (0.,0.,0.),
    (0.,0.,0.)),

    "corr_c" =>
    ((-0.0646774,-0.9520876,-0.6260979),
    (0.1975882,2.9924258,1.2924686),
    (-0.8087562,-2.3802636,1.6542783),
    (0.6902849,-0.2701261,-3.4396744),
    (0.,0.,0.))
)