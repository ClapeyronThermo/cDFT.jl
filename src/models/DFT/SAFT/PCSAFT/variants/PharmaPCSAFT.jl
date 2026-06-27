import Clapeyron: pharmaPCSAFTModel, Δσh20, water08_k

function preallocate_params(system::DFTSystem{<:pharmaPCSAFTModel})
    params_base, nc = invoke(preallocate_params, Tuple{DFTSystem{<:PCSAFTModel}}, system)

    model   = system.model
    backend = system.options.device
    T       = system.structure.conditions[2]

    k  = Int(water08_k(model))
    nc_model = length(model)
    sigma_eff = copy(model.params.sigma.values)

    if k > 0
        Δσ = Δσh20(T)
        for i in 1:nc_model, j in 1:nc_model
            sigma_eff[i,j] += (0.5*(k == i) + 0.5*(k == j)) * Δσ
        end
    end

    # pharmaPCSAFT stores raw LB cross-terms (no k_ij baked in); Clapeyron's m2ϵσ3
    # applies (1 - k0[i,j] - k1[i,j]*T) at runtime. Pre-apply here so f_disp is correct.
    epsilon_eff = copy(model.params.epsilon.values)
    k0_mat = model.params.k.values
    k1_mat = model.params.kT.values
    for i in 1:nc_model
        for j in 1:nc_model
            i == j && continue
            epsilon_eff[i,j] *= (1.0 - k0_mat[i,j] - k1_mat[i,j]*T)
        end
    end

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        eps_vals = model.params.epsilon_assoc.values
        sig3_eff = [sigma_eff[eps_vals.outer_indices[idx]...]^3
                    for idx in 1:length(eps_vals.values)]
        return merge(params_base, (;
            sigma      = Adapt.adapt(backend, sigma_eff),
            epsilon    = Adapt.adapt(backend, epsilon_eff),
            assoc_sig3 = Adapt.adapt(backend, sig3_eff),
        )), nc
    else
        return merge(params_base, (;
            sigma   = Adapt.adapt(backend, sigma_eff),
            epsilon = Adapt.adapt(backend, epsilon_eff),
        )), nc
    end
end