function converge_profile!(model,ρ,T,z;damping=0.05)
    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    function obj(model,ln_G,ρ,T,z,ln_x,μ_res,ρl,α)
        ln_x = reshape(ln_x, (length(z), length(ρ)))
        ln_Gx = reshape(ln_G, (length(z), length(ρ)))
        ln_ρl = log.(ρl)
        for i in @comps
            ρ[i] = update_profile!(ρ[i], exp.(@view(ln_x[:,i])))
        end

        δfδρ_res = δFδρ_res(model,ρ,T,z)
        for i in @comps
            ln_xi = @view(ln_x[:,i])
            δfδρ_resi = @view(δfδρ_res[:,i])
            ln_Gx[:,i] .= (1 .- α) .* ln_xi .+ α .* (ln_ρl[i].+(μ_res[i] .- δfδρ_resi))
        end
        
        return ln_G
    end

    ln_X0 = zeros(length(z),length(ρ))
    ln_GX0 = copy(ln_X0)
    fX(ln_x) = obj(model, copy(ln_x), ρ, T, z, ln_x, μ_res, ρl, damping)
    
    for i in @comps
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    # ρ_new = Solvers.fixpoint(f!,X0,AndersonFixPoint(memory =50),rtol = 1e-4)

    r = fixed_point(fX, ln_X0;Algorithm = :Anderson, 
                                ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)/damping),
                                ConvergenceMetricThreshold=1e-5,
                                MaxIter=10000,
                                MaxM=25)
    
    if ismissing(r.FixedPoint_)
        @warn "Convergence failed"
        ρ_new = exp.(r.Outputs_[:,end])
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    else
        ρ_new = exp.(r.FixedPoint_)
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    end
end

function converge_profile!(
        model,ρ::SphericalDensityProfile,T,z;
        damping=0.05,
        interface_idx::Int64
    )

    len = length(z)
    if 1 > interface_idx || interface_idx > len
        error("Interface index is outside of the bounds.")
    elseif interface_loc === nothing
        interface_idx = round(Int, len/2)
    end

    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    function obj(model,ln_G,ρ,T,z,ln_x,μ_res,ρl,α,idx_intf)
        ln_x = reshape(ln_x, (length(z), length(ρ)))
        ln_Gx = reshape(ln_G, (length(z), length(ρ)))
        ln_ρl = log.(ρl)

        for i in @comps
            ρ_new = exp.(@view(ln_x[:,i]))
            ρ_lb = ρ_new[1] # bound condition at 1
            ρ_ub = ρ_new[end] # boundary condition at end

            # flag
            idx_gds = _find_idx_gibbs_dividing_surface(ρ_new, z)

            Δidx = idx_gds - interface_idx
            if Δidx > 0
                ρ_new = vcat(ρ_new[Δidx+1:end], ρ_ub*ones(Δidx))
            elseif Δidx < 0
                ρ_new = vcat(ρ_lb*ones(-Δidx), ρ_new[1:end+Δidx])
            end # else don't do anything to ρ_new
            ρ[i] = update_profile!(ρ[i], ρ_new)
        end

        δfδρ_res = δFδρ_res(model,ρ,T,z)
        for i in @comps
            ln_xi = @view(ln_x[:,i])
            δfδρ_resi = @view(δfδρ_res[:,i])
            ln_Gx[:,i] .= (1 .- α) .* ln_xi .+ α .* (ln_ρl[i].+(μ_res[i] .- δfδρ_resi))
        end
        
        return ln_G
    end

    ln_X0 = zeros(length(z),length(ρ))
    ln_GX0 = copy(ln_X0)
    fX(ln_x) = obj(model, copy(ln_x), ρ, T, z, ln_x, μ_res, ρl, damping, interface_idx)
    
    for i in @comps
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    # ρ_new = Solvers.fixpoint(f!,X0,AndersonFixPoint(memory =50),rtol = 1e-4)

    r = fixed_point(fX, ln_X0;Algorithm = :Anderson, 
                                ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)/damping),
                                ConvergenceMetricThreshold=1e-5,
                                MaxIter=10000,
                                MaxM=25)
    
    if ismissing(r.FixedPoint_)
        @warn "Convergence failed"
        ρ_new = exp.(r.Outputs_[:,end])
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    else
        ρ_new = exp.(r.FixedPoint_)
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    end
end

# assumes the surface excess is zero.
# uses trapezoidal area
# for cartesian profiles, for now
# for now, assumes monotonically decreasing density function
# TODO: add support for non-monotonic density function
function _find_z_gibbs_dividing_surface(ρ_new, z)
    area_left = 0.
    area_right = 0.
    
    left = 1
    right = length(z)

    # ρ bulk
    ρb_left = ρ_new[1]
    ρb_right = ρ_new[end]

    while true
        if abs(area_left) < abs(area_right)
            left += 1
            area_left += (z[left] - z[left-1])/2 * (2 * ρb_left - ρ_new[left] - ρ_new[left-1])
        else
            right -= 1
            area_right += (z[right+1] - z[right])/2 * (ρ_new[right] + ρ_new[right+1] - 2 * ρb_right)
        end
        if left == right
            if abs(area_left) < abs(area_right)
                right += 1
            else
                left -= 1
            end
            break
        end
    end

    # now the target z is somewhere in between z[left] and z[right]
    # use linear interpolation to find the exact z
    ΔA = area_left - area_right
    z_gds = _calc_z_gds(ΔA, z[right], z[left], ρb_right, ρb_left, ρ_new[right], ρ_new[left])
    return z_gds
end

function _calc_z_gds(ΔA, zr, zl, ρbr, ρbl, ρr, ρl)
    m = (ρr-ρl)/(zr-zl)
    term1 = ρl*(zr-zl)
    term2 = (ρr-2*ρbr)*zr
    term3 = (2*ρbl-ρl)*zl
    term4 = m*zl*(zl-zr)
    denom = m*(zr-zl) - ρr + ρl + 2*ρbr - 2*ρbl
    numer = 2*ΔA - term1 - term2 - term3 - term4
    return numer/denom
end