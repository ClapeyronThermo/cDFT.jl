struct AndersonFixPoint{T<:Real} <: Solvers.AbstractFixPoint
    delay::Int64
    memory::Int64
    damping::T
    picard_damping::T
    drop_tol::T         # max allowed condition number (κ); Inf => no SVD conditioning
    verbose::Bool
end

AndersonFixPoint(;picard_damping=1e-8,damping=1e-8,memory=50,delay=10,drop_tol=Inf, verbose=false) =
    AndersonFixPoint(delay,memory,damping,picard_damping,drop_tol,verbose)

function Solvers.promote_method(method::AndersonFixPoint,T)
    AndersonFixPoint(method.delay,method.memory,T(method.damping),T(method.picard_damping),T(method.drop_tol),method.verbose)
end

rtol_anderson(output,input) = maximum(x->(abs(first(x) - last(x))),zip(output,input))/maximum(output)


function Solvers._fixpoint(f::F,
    x0::X where {X <:AbstractVector{T}},
    method::AndersonFixPoint,
    atol::T = zero(T),
    rtol::T = 8*eps(T),
    max_iters::Int=10000,
    return_last::Bool=false) where {F,T<:Real}

    # ---- unpack ----
    m       = method.memory
    delay   = method.delay
    ω_pic   = method.picard_damping
    ω       = method.damping
    κmax    = method.drop_tol          # interpreted as max allowed cond(RR)
    verbose = method.verbose

    @assert 0 < ω ≤ one(T) "damping must be in (0,1]"
    @assert 0 ≤ delay ≤ max_iters
    @assert m ≥ 1

    # ---- helpers ----
    finite(v) = all(isfinite, v)

    function safe_eval(f, x)
        y = f(x)
        return finite(y) ? y : nothing
    end

    # Backtrack along direction d from (x,fx,r) to get finite, decreasing residual
    function try_backtrack(f, x, fx, r, d, nr; beta=T(0.3), max_bt::Int=10, armijo::T=T(1e-4))
        λ = one(T)
        for _ in 1:max_bt
            x_try = x + λ*d
            fx_try = safe_eval(f, x_try)
            if fx_try !== nothing
                r_try = fx_try - x_try
                if finite(r_try) && norm(r_try) ≤ (one(T) - armijo*λ)*nr
                    return x_try, fx_try, r_try, true
                end
            end
            λ *= beta
        end
        return x, fx, r, false
    end

    # ---- init ----
    x   = copy(x0)
    fx  = safe_eval(f, x)
    if fx === nothing
        if return_last; return x; else error("f(x0) returned NaN/Inf or threw."); end
    end
    r   = fx - x
    r0  = norm(r)
    tol = max(atol, rtol * max(r0, one(T)))
    n   = length(x)

    dR  = zeros(T, n, m)   # residual diffs
    dX  = zeros(T, n, m)   # iterate  diffs
    used = 0
    k = 0

    # utility: push one (Δr, Δx) pair into history (newest at column 1)
    function push_memory!(Δr::AbstractVector{T}, Δx::AbstractVector{T})
        used = min(used + 1, m)
        if used > 1
            @views dR[:, 2:used] .= dR[:, 1:used-1]
            @views dX[:, 2:used] .= dX[:, 1:used-1]
        end
        @views dR[:, 1] .= Δr
        @views dX[:, 1] .= Δx
    end

    # ---------------- prep: damped Picard (and initialize memory) ----------------
    while k < delay && norm(r) > tol
        k += 1
        x_old, r_old = x, r

        # candidate damped Picard step
        x_cand = (one(T)-ω_pic)*x + ω_pic*fx
        d_step = x_cand - x
        fx_cand = safe_eval(f, x_cand)

        accepted = false
        if fx_cand !== nothing
            r_cand = fx_cand - x_cand
            if finite(r_cand)
                x, fx, r = x_cand, fx_cand, r_cand
                accepted = true
            end
        end
        if !accepted
            # backtrack from x along d_step
            x_bt, fx_bt, r_bt, ok = try_backtrack(f, x, fx, r, d_step, norm(r))
            if ok
                x, fx, r = x_bt, fx_bt, r_bt
                accepted = true
                verbose && println("prep $k  (backtracked)  ‖r‖=$(norm(r))")
            else
                # couldn’t find a finite/decreasing step → stop prepping
                verbose && println("prep $k  failed; stopping prep")
                break
            end
        end

        # use accepted prep step to INITIALIZE MEMORY
        push_memory!(r - r_old, x - x_old)
        verbose && println("prep $k  ‖r‖=$(norm(r))  (mem=$used)")
    end
    if norm(r) ≤ tol
        return x
    end

    # ---------------- Anderson stage ----------------
    while k < max_iters && norm(r) > tol
        k += 1
        x_old, r_old, nr_old = x, r, norm(r)

        # ---- fallback: damped Picard (NaN-safe) ----
        x_cand = (one(T)-ω)*x + ω*fx
        d_fallback = x_cand - x
        fx_cand = safe_eval(f, x_cand)

        accepted_fallback = false
        if fx_cand !== nothing
            r_cand = fx_cand - x_cand
            if finite(r_cand)
                x, fx, r = x_cand, fx_cand, r_cand
                accepted_fallback = true
                verbose && println("iter $k  fallback  ‖r‖=$(norm(r))")
            end
        end
        if !accepted_fallback
            x, fx, r, ok = try_backtrack(f, x, fx, r, d_fallback, nr_old)
            if ok
                verbose && println("iter $k  fallback(backtracked)  ‖r‖=$(norm(r))")
            else
                # hard reset memory and continue cautiously
                used = 0; fill!(dR, zero(T)); fill!(dX, zero(T))
                verbose && println("iter $k  fallback failed; history reset")
                continue
            end
        end

        # ---- update memory with the just-accepted fallback step ----
        push_memory!(r - r_old, x - x_old)

        # ---- Anderson step (with κ cap + NaN-safe backtracking) ----
        if used ≥ 1
            RR = @views dR[:, 1:used]
            DX = @views dX[:, 1:used]

            # If memory got polluted, reset and skip Anderson this round
            if !(finite(RR) && finite(DX))
                used = 0; fill!(dR, zero(T)); fill!(dX, zero(T))
                verbose && println("iter $k  memory not finite; reset")
                continue
            end

            U, S, Vt = svd(RR; full=false)
            σ = S
            if !isempty(σ)
                keep_idx = if isfinite(κmax)
                    σ1 = maximum(σ); thresh = σ1 / κmax
                    idx = findall(σ .≥ thresh)
                    isempty(idx) ? [argmax(σ)] : idx
                else
                    collect(1:length(σ))
                end
                rnk = length(keep_idx)

                RR_eff = U[:, keep_idx] * Diagonal(σ[keep_idx])   # n×rnk
                DX_eff = DX * (Vt[:, keep_idx])                   # n×rnk
                γ = Diagonal(σ[keep_idx] .+ eps(T)) \ (U[:, keep_idx]' * r)

                x_try = x - DX_eff * γ
                d_and = x_try - x
                fx_try = safe_eval(f, x_try)

                # Decide accept/line-search
                accept = false
                if fx_try !== nothing
                    r_try = fx_try - x_try
                    if finite(r_try) && norm(r_try) ≤ nr_old
                        x, fx, r = x_try, fx_try, r_try
                        accept = true
                        # overwrite memory with rotated, truncated basis (modifies memory)
                        @views dR[:, 1:rnk] .= RR_eff
                        @views dX[:, 1:rnk] .= DX_eff
                        used = rnk
                        verbose && println("iter $k  Anderson  ‖r‖=$(norm(r))  (r=$rnk)")
                    end
                end

                if !accept
                    # Backtrack along Anderson direction
                    x_bt, fx_bt, r_bt, ok = try_backtrack(f, x, fx, r, d_and, norm(r))
                    if ok
                        x, fx, r = x_bt, fx_bt, r_bt
                        verbose && println("iter $k  Anderson(backtracked)  ‖r‖=$(norm(r))")
                    else
                        verbose && println("iter $k  Anderson rejected; kept fallback")
                    end
                end
            end
        end
    end

    # ---- exit ----
    if norm(r) ≤ tol
        return x
    elseif return_last
        return x
    else
        error("Anderson did not converge in $max_iters iterations. Final ‖r‖=$(norm(r)); tol=$tol.")
    end
end