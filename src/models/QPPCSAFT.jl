using Clapeyron: QPCPSAFTModel

function f_res(system::DFTSystem, model::QPCPSAFTModel,n)
    return f_hs(system,model,n[2,:],n[3,:],n[4,:]) + f_hc(system,model,n[1,:],n[5,:],n[6,:]) + f_disp(system,model,n[7,:]) + f_polar(system,model,n[7,:]) + f_assoc(system,model,n[2,:],n[3,:],n[4,:])
end


function f_polar(system::DFTSystem, model::QPCPSAFTModel, ρ̄)
    species = system.species
  (_, T, _) = system.structure.conditions
  μ̄² = model.params.dipole2.values
  Q̄² = model.params.quadrupole2.values
  has_dp = !all(iszero, μ̄²)
  has_qp = !all(iszero, Q̄²)
  if !has_dp && !has_qp return zero(T+first(ρ̄)) end

  ψ = 1.3862
  HSd = [species[i].size[1] for i in @comps]
  m = model.params.segment.values
  ϵ = model.params.epsilon.values
  σ = model.params.sigma.values

  ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
  η = π/6*@sum(ρ̄[i]*m[i]*HSd[i]^3)
  ∑ρ̄ = sum(ρ̄)
  x = ρ̄ /∑ρ̄
  nc = length(model)

  a_mp_total = zero(T+ρ̄)
  a_mp_total += has_dp && a_dd(x,m,ϵ,σ,μ̄²,Q̄²,η,∑ρ̄,T,nc)
  a_mp_total += has_qp && a_qq(x,m,ϵ,σ,μ̄²,Q̄²,η,∑ρ̄,T,nc)
  a_mp_total += has_dp && has_qp && a_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,∑ρ̄,T,nc)

  return ∑ρ̄*a_mp_total
end

function a_polar(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
  A₂ = A2(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
  iszero(A₂) && return zero(A₂)
  A₃ = A3(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
  return A₂^2/(A₂-A₃)
end
function a_dd(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return a_polar(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,:DD)
end
function a_qq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return a_polar(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,:QQ)
end
function a_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    A₂ = A2_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    iszero(A₂) && return zero(A₂)
    A₃ = A3_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return A₂^2/(A₂-A₃)
end

function A2(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
  dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
  
  P̄² = []
  p_comps = []
  p = 0
  coeff = 0.
  if type == :DD
      if isempty(dp_comps) return 0. end
      P̄² = μ̄²
      p_comps = dp_comps
      p = 3
      coeff = 1.
  end
  if type == :QQ
      if isempty(qp_comps) return 0. end
      P̄² = Q̄²
      p_comps = qp_comps
      p = 7
      coeff = 9/16
  end
  _a_2 = zero(T+first(x))
  @inbounds for (idx, i) ∈ enumerate(p_comps)
      _J2_ii = J2(m[i],m[i],ϵ[i,i],η,T,type)
      xᵢ = x[i]
      P̄²ᵢ = P̄²[i]
      _a_2 +=xᵢ^2*P̄²ᵢ^2/σ[i,i]^p*_J2_ii
      for j ∈ p_comps[idx+1:end]
          _J2_ij = J2(m[i],m[j],ϵ[i,j],η,T,type)
          _a_2 += 2*xᵢ*x[j]*P̄²ᵢ*P̄²[j]/σ[i,j]^p*_J2_ij
      end
  end
  _a_2 *= -π*coeff*ρ̄/T^2
  return _a_2
end

function A2_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
  dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
  _a_2 = zero(T+first(x))
  @inbounds for i in dp_comps
      for j ∈ qp_comps
          _J2_ij = J2(m[i],m[j],ϵ[i,j],η,T,:DQ)
          _a_2 += x[i]*x[j]*μ̄²[i]*Q̄²[j]/σ[i,j]^5*_J2_ij
      end
  end
  _a_2 *= -π*9/4*ρ̄/T^2
  return _a_2
end

function A3(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
  dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
  P̄² = []
  p_comps = []
  p = 0
  coeff = 0.
  _0 = zero(T+first(x))
  if type == :DD
      if isempty(dp_comps) return _0 end
      P̄² = μ̄²
      p_comps = dp_comps
      p = 1
      coeff = -4*π^2/3
  end
  if type == :QQ
      if isempty(qp_comps) return _0 end
      P̄² = Q̄²
      p_comps = qp_comps
      p = 3
      coeff = 9*π^2/16
  end

  _a_3 = _0
  @inbounds for (idx_i,i) ∈ enumerate(p_comps)
      _J3_iii = J3(m[i],m[i],m[i],η,type)
      xᵢ,P̄ᵢ² = x[i],P̄²[i]
      a_3_i = xᵢ*P̄ᵢ²/σ[i,i]^p
      _a_3 += a_3_i^3*_J3_iii
      for (idx_j,j) ∈ enumerate(p_comps[idx_i+1:end])
          xⱼ,P̄ⱼ² = x[j],P̄²[j]
          σij⁻ᵖ = 1/σ[i,j]^p
          a_3_iij = xᵢ*P̄ᵢ²*σij⁻ᵖ
          a_3_ijj = xⱼ*P̄ⱼ²*σij⁻ᵖ
          a_3_j = xⱼ*P̄ⱼ²/σ[j,j]^p
          _J3_iij = J3(m[i],m[i],m[j],η,type)
          _J3_ijj = J3(m[i],m[j],m[j],η,type)
          _a_3 += 3*a_3_iij*a_3_ijj*(a_3_i*_J3_iij + a_3_j*_J3_ijj)
          for k ∈ p_comps[idx_i+idx_j+1:end]
              xₖ,P̄ₖ² = x[k],P̄²[k]
              _J3_ijk = J3(m[i],m[j],m[k],η,type)
              _a_3 += 6*xᵢ*xⱼ*xₖ*P̄ᵢ²*P̄ⱼ²*P̄ₖ²*σij⁻ᵖ/(σ[i,k]*σ[j,k])^p*_J3_ijk
          end
      end
  end
  _a_3 *= coeff*ρ̄^2/T^3
  return _a_3
end

function A3_dq(x,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
  dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
  _a_3 = zero(T+first(x))
  @inbounds for i ∈ dp_comps
      for j ∈ union(dp_comps, qp_comps)
          for k ∈ qp_comps
              _J3_ijk = J3(m[i],m[j],m[k],η,:DQ)
              _a_3 += x[i]*x[j]*x[k]*σ[i,i]/
                  (σ[k,k]*(σ[i,j]*σ[i,k]*σ[j,k])^2)*
                  μ̄²[i]*Q̄²[k]*(σ[j,j]*μ̄²[j]+1.19374/σ[j,j]*Q̄²[j])*_J3_ijk
          end
      end
  end
  _a_3 *= -ρ̄^2/T^3
  return _a_3
end

# Returns [Dipole Comp idxs], [Quadrupole Comp idxs]
function polar_comps(μ̄²,Q̄²,nc)
  dipole_comps = []
  quadrupole_comps = []
  for i in 1:nc
      if !iszero(μ̄²[i]) push!(dipole_comps,i) end
      if !iszero(Q̄²[i]) push!(quadrupole_comps,i) end
  end
  return dipole_comps, quadrupole_comps
end

function J2(mᵢ,mⱼ,ϵᵢⱼ,η,T,type::Symbol)
  corr_consts = NamedTuple()
  ϵᵢⱼT⁻¹ = ϵᵢⱼ/T
  m̄ = sqrt(mᵢ*mⱼ)
  n_range = 0:4

  if type == :DD
      corr_consts = DD_consts
      m̄ = minimum([m̄, 2.0])
  elseif type == :DQ
      corr_consts = DQ_consts
      n_range = 0:3 # Needs revision
  elseif type == :QQ
      corr_consts = QQ_consts
  end

  m1 = 1. - 1/m̄
  m2 = m1 * (1. - 2/m̄)
  corr_a = corr_consts[:corr_a]
  corr_b = corr_consts[:corr_b]

  J_2ij = zero(η)

  for n ∈ n_range
      a0, a1, a2 = corr_a[n+1]
      b0, b1, b2 = corr_b[n+1]
      a_nij = a0 + a1*m1 + a2*m2
      b_nij = b0 + b1*m1 + b2*m2
      J_2ij += (a_nij + b_nij*ϵᵢⱼT⁻¹) * η^n
  end

  return J_2ij
end

function J3(mᵢ,mⱼ,mₖ,η,type::Symbol)
  m̄ = cbrt(mᵢ*mⱼ*mₖ)
  corr_c = ()
  m1 = 1. - 1/m̄
  m2 = 0.
  n_range = 0:4

  if type == :DD
      corr_c = DD_consts[:corr_c]
      m̄ = minimum([m̄, 2.0])
      m2 = m1 * (1. - 2/m̄)
  elseif type == :DQ
      corr_c = DQ_consts[:corr_c]
      n_range = 0:3
  elseif type == :QQ
      corr_c = QQ_consts[:corr_c]
      m2 = m1 * (1. - 2/m̄)
  end

  J_3ijk = zero(η)
  if type == :DQ
      for n ∈ n_range
          c0, c1 = corr_c[n+1]
          c_nijk = c0 + c1*m1
          J_3ijk += c_nijk*η^n
      end
  else
      for n ∈ n_range
          c0, c1, c2 = corr_c[n+1]
          c_nijk = c0 + c1*m1 + c2*m2
          J_3ijk += c_nijk*η^n
      end
  end

  return J_3ijk
end

const QQ_consts = (
    corr_a = 
    ((1.2378308, 1.2854109,	1.7942954),
    (2.4355031,	-11.465615,	0.7695103),
    (1.6330905,	22.086893,	7.2647923),
    (-1.6118152, 7.4691383,	94.486699),
    (6.9771185,	-17.197772,	-77.148458)),

    corr_b = 
    ((0.4542718, -0.813734, 6.8682675),
    (-4.5016264, 10.06403, -5.1732238),
    (3.5858868,	-10.876631, -17.240207),
    (0., 0., 0.),
    (0., 0., 0.)),

    corr_c =
    ((-0.5000437, 2.0002094, 3.1358271),
    (6.5318692, -6.7838658, 7.2475888),
    (-16.01478, 20.383246, 3.0759478),
    (14.42597, -10.895984, 0.),
    (0., 0., 0.))    
)

const DQ_consts = (
    corr_a =
    ((0.697095, -0.6734593, 0.6703408),
    (-0.6335541, -1.4258991, -4.3384718),
    (2.945509, 4.1944139, 7.2341684),
    (-1.4670273, 1.0266216, 0.)),

    corr_b =
    ((-0.4840383, 0.6765101, -1.1675601),
    (1.9704055, -3.0138675, 2.1348843),
    (-2.1185727, 0.4674266, 0.),
    (0.,	0.,	0.)),

    corr_c =
    ((0.7846431, -2.072202),
    (3.3427, -5.863904),
    (0.4689111, -0.1764887),
    (0., 0.))
)