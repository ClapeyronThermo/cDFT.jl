using Clapeyron, Enzyme, StaticArrays, CUDA, Adapt, KernelAbstractions, LinearAlgebra

# Constants for PCSAFT dispersion
const PCSAFT_CORR1 = (
    (0.9105631445, -0.3084016918, -0.0906148351),
    (0.6361281449, 0.1860531159, 0.4527842806),
    (2.6861347891, -2.5030047259, 0.5962700728),
    (-26.547362491, 21.419731138, -1.7241829131),
    (97.759208784, -65.255885330, -4.1302112531),
    (-159.59154066, 83.318680481, 13.776631870),
    (91.297774084, -33.746922930, -8.6728470368)
)
const PCSAFT_CORR2 = (
    (0.7240946941, -0.5754482788, 0.0976883116),
    (2.2382791861, 0.6995095521, -0.2557538506),
    (-4.0025849485, 3.8925673390, -9.1558561530),
    (-21.003576815, -17.215471648, 20.642075971),
    (26.855641363, 192.67226447, -38.804430052),
    (206.55133840, -161.82646165, 93.626774077),
    (-355.60237127, -165.20769341, -29.666905585)
)

@inline function I_lite(corr, m̄, η)
    res = 0.0
    m2 = (m̄ - 1.0) / m̄
    m3 = m2 * (m̄ - 2.0) / m̄
    c1 = corr[1]; res += (c1[1] + m2 * c1[2] + m3 * c1[3])
    c2 = corr[2]; res += (c2[1] + m2 * c2[2] + m3 * c2[3]) * η
    c3 = corr[3]; res += (c3[1] + m2 * c3[2] + m3 * c3[3]) * η * η
    c4 = corr[4]; res += (c4[1] + m2 * c4[2] + m3 * c4[3]) * η * η * η
    c5 = corr[5]; res += (c5[1] + m2 * c5[2] + m3 * c5[3]) * η * η * η * η
    c6 = corr[6]; res += (c6[1] + m2 * c6[2] + m3 * c6[3]) * η * η * η * η * η
    c7 = corr[7]; res += (c7[1] + m2 * c7[2] + m3 * c7[3]) * η * η * η * η * η * η
    return res
end

@inline function f_res_lite_void_global(out, n, HSd, m, σ, ϵ, T, k, ::Val{NC}, ::Val{ND}) where {NC, ND}
    _pi = 3.141592653589793
    n₀ = 0.0; n₁ = 0.0; n₂ = 0.0; n₃₃ = 0.0
    nv1_1 = 0.0; nv1_2 = 0.0; nv1_3 = 0.0
    nv2_1 = 0.0; nv2_2 = 0.0; nv2_3 = 0.0
    @inbounds for i in 1:NC
        mi = m[i]; HSdi = HSd[i]; ni_2 = n[2, i, k]; nim = ni_2 * mi
        n₀ += nim / HSdi; n₁ += 0.5 * nim; n₂ += _pi * nim * HSdi
        if ND >= 1; nvi = n[4, i, k]; nv1_1 += nvi * mi / HSdi; nv2_1 += -2.0 * _pi * nvi * mi; end
        if ND >= 2; nvi = n[5, i, k]; nv1_2 += nvi * mi / HSdi; nv2_2 += -2.0 * _pi * nvi * mi; end
        if ND >= 3; nvi = n[6, i, k]; nv1_3 += nvi * mi / HSdi; nv2_3 += -2.0 * _pi * nvi * mi; end
        n₃₃ += n[3, i, k] * mi
    end
    n_v1_dot_v2 = nv1_1 * nv2_1 + nv1_2 * nv2_2 + nv1_3 * nv2_3
    n_v2_sq = nv2_1 * nv2_1 + nv2_2 * nv2_2 + nv2_3 * nv2_3
    eps_val = 1e-15; denom_log = 1.0 - n₃₃; log_1_n33 = Base.log(abs(denom_log) + eps_val); inv_1_n₃₃ = 1.0 / (denom_log + eps_val)
    denom_f_hs = 12.0 * _pi * (n₃₃ * n₃₃ + eps_val)
    res_f_hs = -n₀ * log_1_n33 + (n₁ * n₂ - n_v1_dot_v2) * inv_1_n₃₃ + (n₂^3 / 3.0 - n₂ * n_v2_sq) * (log_1_n33 / denom_f_hs + 1.0 / (denom_f_hs * (denom_log * denom_log + eps_val)))
    ζ₃ = 0.0; ζ₂ = 0.0; idx_n6 = 4 + ND + 1
    @inbounds for i in 1:NC; mi = m[i]; ρ̄hci = n[idx_n6, i, k]; HSdi = HSd[i]; ζ₃ += mi * ρ̄hci; ζ₂ += mi * ρ̄hci / HSdi; end
    ζ₃ *= 0.125; ζ₂ *= 0.125; inv_1_ζ₃ = 1.0 / (1.0 - ζ₃ + eps_val); res_f_hc = 0.0; idx_n5 = 4 + ND
    @inbounds for i in 1:NC
        λ = n[idx_n5, i, k] / (2.0 * HSd[i]); yᵈᵈ = inv_1_ζ₃ + 1.5 * HSd[i] * ζ₂ * inv_1_ζ₃^2 + 0.5 * HSd[i]^2 * ζ₂^2 * inv_1_ζ₃^3
        res_f_hc += -n[1, i, k] * (m[i] - 1.0) * Base.log(abs(yᵈᵈ * λ / (n[1, i, k] + eps_val)) + eps_val)
    end
    ψ = 1.3862; ρ̄z_sum = eps_val; m̄_top = 0.0; η_disp = 0.0; factor = 3.0 / (4.0 * ψ^3 * _pi); idx_n7 = 4 + ND + 2
    @inbounds for i in 1:NC; ρ̄zi = n[idx_n7, i, k] * factor / (HSd[i]^3); ρ̄z_sum += ρ̄zi; m̄_top += ρ̄zi * m[i]; η_disp += m[i] * ρ̄zi * HSd[i]^3; end
    m̄ = m̄_top / ρ̄z_sum; η_disp *= _pi / 6.0; m2ϵσ3_1 = 0.0; m2ϵσ3_2 = 0.0
    @inbounds for i in 1:NC
        ρi = n[idx_n7, i, k] * factor / (HSd[i]^3)
        @inbounds for j in i:NC
            ρj = n[idx_n7, j, k] * factor / (HSd[j]^3); const_ij = ρi * ρj * m[i] * m[j] * σ[i,j]^3; eps_T = ϵ[i,j] / (T + eps_val)
            if i == j; m2ϵσ3_1 += const_ij * eps_T; m2ϵσ3_2 += const_ij * eps_T^2
            else; m2ϵσ3_1 += 2.0 * const_ij * eps_T; m2ϵσ3_2 += 2.0 * const_ij * eps_T^2
            end
        end
    end
    ηd = η_disp; ηd2 = ηd * ηd; ηd4 = ηd2 * ηd2; inv_1_ηd = 1.0 / (1.0 - ηd + eps_val); inv_2_ηd = 1.0 / (2.0 - ηd + eps_val)
    C₁ = 1.0 + m̄ * (8.0 * ηd - 2.0 * ηd2) / (ηd4 + eps_val) + (1.0 - m̄) * (20.0 * ηd - 27.0 * ηd2 + 12.0 * (ηd2 * ηd) - 2.0 * ηd4) * inv_1_ηd * inv_1_ηd * inv_2_ηd * inv_2_ηd
    I₁ = I_lite(PCSAFT_CORR1, m̄, ηd); I₂ = I_lite(PCSAFT_CORR2, m̄, ηd)
    res_f_disp = -2.0 * _pi * I₁ * m2ϵσ3_1 - _pi * m̄ * I₂ * m2ϵσ3_2 / (C₁ + eps_val)
    out[k] = res_f_hs + res_f_hc + res_f_disp
    return nothing
end

@kernel function test_kernel!(dn, out, dout, n, @Const(HSd), @Const(m), @Const(sigma), @Const(epsilon), T, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}) where {NF, NB, NC, ND}
    k = @index(Global)
    # Global Void pattern: Differentiate f_res_lite_void_global wrt global memory n
    Enzyme.autodiff_deferred(Reverse, Const(f_res_lite_void_global), Const,
        Duplicated(out, dout),
        Duplicated(n, dn),
        Const(HSd), Const(m), Const(sigma), Const(epsilon),
        Const(Float64(T)), Const(k), Const(Val(NC)), Const(Val(ND)))
end

function test_enzyme_gpu()
    if !CUDA.functional(); println("CUDA not functional"); return; end
    backend = CUDABackend(); model = PCSAFT(["hexane"]); T = 298.15; NC = length(model); ND = 1; NF = 7; NB = 1 
    m = model.params.segment.values; sigma = model.params.sigma.values; epsilon = model.params.epsilon.values; HSd = [3.702] 
    m_gpu = Adapt.adapt(backend, m); sigma_gpu = Adapt.adapt(backend, sigma); epsilon_gpu = Adapt.adapt(backend, epsilon); HSd_gpu = Adapt.adapt(backend, HSd)
    
    # n must be (NF, NC, NB) to match the global void pattern indexing
    n_cpu = fill(0.1, NF, NC, NB); n_gpu = Adapt.adapt(backend, n_cpu)
    dn_gpu = KernelAbstractions.allocate(backend, Float64, NF, NC, NB); fill!(dn_gpu, 0.0)
    out_gpu = KernelAbstractions.allocate(backend, Float64, NB); fill!(out_gpu, 0.0)
    dout_gpu = KernelAbstractions.allocate(backend, Float64, NB); fill!(dout_gpu, 1.0)

    println("Launching Enzyme kernel on GPU (Full logic, Global Void pattern)...")
    try
        kernel = test_kernel!(backend)
        kernel(dn_gpu, out_gpu, dout_gpu, n_gpu, HSd_gpu, m_gpu, sigma_gpu, epsilon_gpu, Float64(T), Val(NF), Val(NB), Val(NC), Val(ND), ndrange=(NB,))
        KernelAbstractions.synchronize(backend)
        println("Gradients obtained (first batch):")
        display(Array(dn_gpu)[:, :, 1])
    catch e
        println("Caught error during GPU compilation/execution:")
        showerror(stdout, e)
    end
end

test_enzyme_gpu()
