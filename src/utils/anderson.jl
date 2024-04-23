struct AndersonFixPoint{T<:Real} <: Solvers.AbstractFixPoint
    delay::Int64
    memory::Int64
    damping::T
    picard_damping::T
    drop_tol::T
end

AndersonFixPoint(;picard_damping=1e-2,damping=1e-2,memory=50,delay=100,drop_tol=Inf) = AndersonFixPoint(delay,memory,damping,picard_damping,drop_tol)

function Solvers.promote_method(method::AndersonFixPoint,T)
    return AndersonFixPoint(method.delay,method.memory,T(method.damping),T(method.picard_damping),T(method.drop_tol))
end

rtol_anderson(output,input) = maximum(x->(abs(first(x)/last(x)) -1),zip(output,input))
# args
function Solvers._fixpoint(f::F,
    x0::X where {X <:AbstractVector{T}},
    method::AndersonFixPoint,
    atol::T = zero(T),
    rtol::T =8*eps(T),
    max_iters=10000,
    return_last=false) where {F,T<:Real}
    #==============================================================
      Do initial function iterations; default is a delay of 0,
      but we always do at least one evaluation of G, to set up AA.
      Notation: AA solves g(x) - x = 0 or F(x) = 0
    ==============================================================#
    m = method.memory
    picard_iter = method.delay
    picard_damping = method.picard_damping
    damping = method.damping

    n = length(x0)
    X = zeros(n, m)
    Fx = zeros(n, m)

    X[:, 1] = x0
    Fx[:, 1] = f(x0)
    for i in 1:picard_iter
        x = X[:, 1] + picard_damping .* (f(X[:, 1]) - X[:, 1])
        fval = f(x)
        X[:,2:min(i+1, m)] = X[:,1:min(i, m-1)]
        Fx[:,2:min(i+1, m)] = Fx[:,1:min(i, m-1)]
        X[:, 1] = x
        Fx[:, 1] = fval

        finite_check = NLSolvers.isallfinite(x)
        if norm(fval-x) < rtol  || !finite_check
            return x
        end
    end

    # Initialize A matrix
    m_eff = min(picard_iter+1, m)
    A = zeros(n, m-1)

    drop_tol = method.drop_tol

    # Iterate until convergence
    for k in picard_iter+1:max_iters
        # Compute residual
        r = Fx[:, 1] - X[:, 1]

        # Compute update
        check_cond = true
        R = zeros(m_eff-1, m_eff-1)
        Q = zeros(n, m_eff-1)
        b = zeros(m_eff-1)
            
        while check_cond
            for i in 2:min(k, m_eff)
                A[:, i-1] = Fx[:, i] - X[:, i]
            end
            A .-= r            
            b = -r
            Q, R = qr(A[:, 1:min(k-1, m_eff-1)])
            # println(R)
            if cond(R)<drop_tol
                check_cond = false
                break
            else
                m_eff -= 1
            end
        end
        R = [R; zeros(n-m_eff+1, m_eff-1)]

        α = R \ (Q' * b)

        Δx = r
        for i in 2:min(k, m_eff)
            Δx .+= α[i-1] .* (X[:, i] - X[:, 1] + A[:, i-1])
        end

        # Update guess
        x = X[:, 1] + damping .* (Δx)
        fval = f(x)
        finite_check = NLSolvers.isallfinite(x)

        # Check for convergence
        if mod(k,10) == 0
            # println(rtol_anderson(fval,x))
        end
        if rtol_anderson(fval,x) < rtol || !finite_check
            # println(k)
            return x
        end

        # Update history
        if m_eff < m
            m_eff += 1
            X[:, 2:m_eff] = X[:, 1:m_eff-1]
            Fx[:, 2:m_eff] = Fx[:, 1:m_eff-1]
            X[:, 1] = x
            Fx[:, 1] = fval
        else
            X[:, 2:end] = X[:, 1:end-1]
            Fx[:, 2:end] = Fx[:, 1:end-1]
            X[:, 1] = x
            Fx[:, 1] = fval
        end
        
    end
    # return X[:,1]
end