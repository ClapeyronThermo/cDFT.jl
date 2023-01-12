struct AndersonFixPoint{T<:Real} <: Solvers.AbstractFixPoint
    delay::Int64
    memory::Int64
    beta::T
    droptol::T
end

AndersonFixPoint(;dampingfactor=1.0,delay = 0, memory = 5,droptol = NaN) = AndersonFixPoint(delay,memory,dampingfactor,droptol)

function Solvers.promote_method(method::AndersonFixPoint,T)
    return AndersonFixPoint(method.delay,method.memory,T(method.beta),T(method.droptol))
end

function vv_shift!(G)
    for i = 1:length(G)-1
        G[i] = G[i+1]
    end
end
rtol_anderson(output,input) = maximum(x->(abs(first(x)/last(x)) -1),zip(output,input))
# args
function Solvers._fixpoint(f!::F,
    x0::X where {X <:AbstractVector{T}},
    method::AndersonFixPoint,
    atol::T = zero(T),
    rtol::T =8*eps(T),
    max_iters=100,
    return_last = false) where {F,T<:Real}
    #==============================================================
      Do initial function iterations; default is a delay of 0,
      but we always do at least one evaluation of G, to set up AA.
      Notation: AA solves g(x) - x = 0 or F(x) = 0
    ==============================================================#
    nan = zero(eltype(x0))/zero(eltype(x0))
    Gx = similar(x0)
    Fx = similar(x0)
    x = copy(x0)
    Gx = f!(Gx, x)
    Fx .= Gx .- x

    x .= Gx
    delay_iter = 0
    while delay_iter < method.delay
        delay_iter += 1
        Gx = f!(Gx, x)
        Fx .= Gx .- x
        x .= Gx
        finite_check = NLSolvers.isallfinite(x)
        if norm(Fx) < atol || rtol_anderson(Gx,x) < rtol || !finite_check
            return x
        end
    end

    #==============================================================
      If we got this far, then the delay was not enough to con-
      verge. However, we now hope to have moved to a region where
      everything is well-behaved, and we start the acceleration.
    ==============================================================#

    n = length(x)
    memory = min(n, method.memory)

    Q = x * x[1:memory]'
    R = x[1:memory] * x[1:memory]'

    #==============================================================
      Start Anderson Acceleration. We use QR updates to add new
      successive changes in G to the system, and once the memory
      is exhausted, we use QR downdates to forget the oldest chan-
      ges we have stored.
    ==============================================================#
    effective_memory = 0
    beta = method.beta

    G = [copy(x) for i = 1:memory]
    Δg = copy(Gx)
    Δf = copy(Fx)

    Gold = copy(Gx)
    iter = 0
    while iter < max_iters
        iter += 1
        Fold = copy(Fx)
        Gx = f!(Gx, x)

        Fx .= Gx .- x
        x .= Gx
        # is this actually needed? I think we can avoid these
        @. Δg = Gx - Gold
        @. Δf = Fx - Fold

        Gold .= Gx
        Fold .= Fx
        effective_memory += 1
        # if we've exhausted the memory, downdate
        if effective_memory > memory
            vv_shift!(G)
            NLSolvers.qrdelete!(Q, R, memory)
            effective_memory -= 1
        end

        # Add the latest change to G
        G[effective_memory] .= Δg

        # QR update
        NLSolvers.qradd!(Q, R, vec(Δf), effective_memory)

        # Create views for the system depending on the effective memory counter
        Qv = view(Q, :, 1:effective_memory)
        Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))
        droptol = method.droptol
        # check condition number
        if (!isnan(droptol))
            while cond(Rv) > droptol && effective_memory > 1
                NLSolvers.qrdelete!(Q, R, effective_memory)

                effective_memory -= 1
                Qv = view(Q, :, 1:effective_memory)
                Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))
            end
        end

        # solve least squares problem
        γv = zeros(effective_memory) #view(γv, 1:m_eff)
        ldiv!(Rv, mul!(γv, Qv', vec(Fx)))

        # update next iterate
        for i = 1:effective_memory
            @. x -= γv[i] * G[i]
        end
        
        if beta != one(beta)
            x .= x .- (1 .- beta) .* (Fx .- Qv * Rv * γv)
        end
        finite_check = NLSolvers.isallfinite(x)

        if norm(Fx) < atol || rtol_anderson(Gx,x) < rtol || !finite_check
            break #return (x = x, Fx = Fx, acc_iter = 0, finite = finite_check)
        end
    end
    !return_last && (x .= nan)
    return x
end