"""
aasol(GFix!, x0, m, Vstore; maxit=20,
      rtol=1.e-10, atol=1.e-10, beta=1.0, pdata=nothing, keepsolhist=false,
      picard_maxit=0, picard_beta=1.0, verbose=false)

C. T. Kelley, 2022. Modified 2024.

Julia code for Anderson acceleration. Nothing fancy.

Solvers fixed point problems x = G(x).

You must allocate storage for the function and fixed point map
history --> in the calling program <-- in the array Vstore.

For an n dimensional problem with Anderson(m), Vstore must have
at least 2m + 4 columns and 3m + 3 is better.  If m=0 (Picard) then
V must have at least 4 columns.

Inputs:

- GFix!: fixed-point map, the ! indicates that GFix! overwrites G, your
    preallocated storage for the function value G=G(xin).
    So G=GFix!(G,xin) or G=GFix!(G,xin,pdata) returns G=G(xin).
    Your GFix function MUST end with --> return G <--. See the example
    in the docstrings.

- x0: Initial iterate. It is a vector of size N.
  You should store it as (N) and design G! to use vectors of size (N).
  If you use (N,1) consistently instead, the solvers may work, but I make
  no guarantees.

- m: depth for Anderson acceleration. m=0 is Picard iteration.

- Vstore: Working storage array. For an n dimensional problem Vstore
  should have at least 3m+3 columns unless you are storage bound. If storage
  is a problem, then you can allocate a minimum of 2m+4 columns. The smaller
  allocation exacts a performance penalty, especially for small problems
  and small values of m. So for Anderson(3), Vstore should be no smaller
  than zeros(N,8) with zeros(N,11) a better choice.

  If m=0, then Vstore needs 4 columns.

Keyword Arguments (kwargs):

maxit: default = 20
  Limit on Anderson iterations (after any Picard warmup).

rtol and atol: default = 1.e-10
  Relative and absolute error tolerances.

beta:
  Anderson mixing parameter. Changes G(x) to (1-beta)x + beta G(x).
  Equivalent to accelerating damped Picard iteration. The history
  vector is the one for the damped fixed point map, not the original
  one. Keep this in mind when comparing results.

pdata:
  Precomputed data for the fixed point map. Things will go better if
  you use this rather than hide the data in global variables.

keepsolhist: default = false
  Set this to true to get the history of the iteration in the output
  tuple. Only turn it on if you have use for the data, which can get
  REALLY LARGE.

picard_maxit: default = 0
  Number of plain Picard iterations to run before handing off to
  Anderson. Useful for stabilising ill-conditioned problems, or when
  Anderson requires the iterates to already be close to the fixed
  point before it is started. Setting this to 0 (the default) recovers
  the original behaviour exactly.

picard_beta: default = 1.0
  Damping factor applied during the Picard warmup phase only.
  Independent of the Anderson `beta`. Changes G(x) to
  (1-picard_beta)*x + picard_beta*G(x) during warmup.

verbose: default = false
  Print a one-line summary for every iteration to stdout. Each line
  shows the phase (Picard warmup or Anderson), the iteration counter,
  the current residual norm, and the running atol and rtol tolerances.
  Example output:

    Anderson  k=  1  |res|= 4.487e-01  atol= 1.000e-10  rtol= 1.000e-10
    Anderson  k=  2  |res|= 2.615e-02  atol= 1.000e-10  rtol= 1.000e-10
    ...
    Converged after 8 iterations.

Output:
- A named tuple (solution, functionval, history, stats, idid, errcode, solhist)
  where

   -- solution = converged result

   -- functionval = G(solution)
      You might want to use functionval as your solution since it's
      a Picard iteration applied to the converged Anderson result. If G
      is a contraction it will be better than the solution.

   -- history = the vector of residual norms (||x-G(x)||) for the full
                iteration, including any Picard warmup steps.

   -- stats = named tuple (condhist, alphanorm) of the history of the
              condition numbers of the optimisation problem and l1 norm
              of the coefficients. Only covers the Anderson phase.

      condhist[k] and alphanorm[k] are the condition number and
      coefficient norm for the optimisation problem that computes
      iteration k+1 from iteration k. Recorded for Anderson iterations
      k=1,... until the final iteration K; not recorded for k=0 or the
      final iteration. If history has length K+1 for iterations 0...K
      then condhist and alphanorm have length K-1.

   -- idid = true if the iteration succeeded and false if not.

   -- errcode = 0  if the iteration succeeded
              = -1 if the initial iterate satisfies the termination criteria
              = -2 if ||residual|| > 1e4 * ||residual_0|| (divergence guard)
              = 10 if no convergence after maxit iterations

   -- solhist:
      This is the entire history of the iteration if keepsolhist=true.
      solhist is an N x K array where N is the length of x and K is the
      number of iterations + 1. Otherwise nothing.

### Examples for aasol

#### Duplicate Table 1 from Toth-Kelley 2015.

The final entries in the condition number and coefficient norm statistics
are never used in the computation and we don't compute them in Julia.
See the docstrings, notebook, and the print book for the story on this.

```jldoctest
julia> function tothk!(G, u)
       G[1]=cos(.5*(u[1]+u[2]))
       G[2]=G[1]+ 1.e-8 * sin(u[1]*u[1])
       return G
       end
tothk! (generic function with 1 method)

julia> u0=ones(2,); m=2; vdim=3*m+3; Vstore = zeros(2, vdim);
julia> aout = aasol(tothk!, u0, m, Vstore; rtol = 1.e-10);
julia> aout.history
8-element Vector{Float64}:
 6.50111e-01
 4.48661e-01
 2.61480e-02
 7.25389e-02
 1.53107e-04
 1.18513e-05
 1.82466e-08
 1.04725e-13

julia> [aout.stats.condhist aout.stats.alphanorm]
6×2 Matrix{Float64}:
 1.00000e+00  1.00000e+00
 2.01556e+10  4.61720e+00
 1.37776e+09  2.15749e+00
 3.61348e+10  1.18377e+00
 2.54948e+11  1.00000e+00
 3.67694e+10  1.00171e+00
```

Now with beta = .5 mixing:

```
julia> bout=aasol(tothk!, u0, m, Vstore; rtol = 1.e-10, beta=.5);
julia> bout.history
7-element Vector{Float64}:
 3.25055e-01
 3.70140e-02
 1.81111e-03
 9.55308e-04
 1.25936e-05
 1.40854e-09
 2.18196e-12
```

#### H-equation example with m=2.

```jldoctest
julia> n=16; x0=ones(n,); Vstore=zeros(n,20); m=2;
julia> hdata=heqinit(x0,.99);
julia> hout=aasol(HeqFix!, x0, m, Vstore; pdata=hdata);
julia> hout.history
12-element Vector{Float64}:
 1.47613e+00
 7.47800e-01
 2.16609e-01
 4.32017e-02
 2.66867e-02
 6.82965e-03
 2.70779e-04
 6.51027e-05
 7.35581e-07
 1.85649e-09
 4.94803e-10
 5.18866e-12
```

#### Picard warmup before Anderson

```jldoctest
julia> u0=ones(2,); m=2; vdim=3*m+3; Vstore=zeros(2,vdim);
julia> aout = aasol(tothk!, u0, m, Vstore; picard_maxit=5, picard_beta=0.5);
```
"""

# Private helper: format one verbose iteration line without Printf.
function _aa_verbose_line(phase::String, k::Int, resnorm, tol)
    fmt(x) = lpad(string(round(x, sigdigits=4)), 11)
    println(rpad(phase, 10), "  k=", lpad(k, 3),
            "  |res|=", fmt(resnorm),
            "  tol=",  fmt(tol))
end

function aasol(
    GFix!,
    x0,
    m,
    Vstore;
    maxit        = 20,
    rtol         = 1.e-10,
    atol         = 1.e-10,
    beta         = 1.0,
    pdata        = nothing,
    keepsolhist  = false,
    picard_maxit = 0,
    picard_beta  = 1.0,
    picard_rtol  = 1e-2,
    picard_atol  = 1e-2,
    verbose      = false,
)
    #
    # Startup
    #
    (sol, gx, df, dg, res, DG, QP, Qd, solhist) =
        Anderson_Init(x0, Vstore, m, maxit + picard_maxit, beta, keepsolhist)
    #
    #   First evaluation
    #
    k = 0
    ~keepsolhist || (@views solhist[:, k+1] .= sol)
    gx = EvalF!(GFix!, gx, sol, pdata)
    (beta == 1.0) || (gx = betafix!(gx, sol, beta))
    copy!(res, gx)
    axpy!(-1.0, sol, res)
    resnorm       = norm(res)
    resnorm_up_bd = 1.e4 * resnorm
    tol           = rtol * resnorm + atol
    picard_tol    = picard_rtol * resnorm + picard_atol
    ItData        = ItStatsA(resnorm)
    toosoon       = (resnorm <= tol)

    if verbose
        _aa_verbose_line("Init", 0, resnorm, tol)
    end

    # ----------------------------------------------------------------
    #  Optional Picard warmup phase
    # ----------------------------------------------------------------
    if ~toosoon && picard_maxit > 0
        # The first G(x0) was already evaluated above; accept that step
        # and then run picard_maxit - 1 further Picard steps.
        copy!(sol, gx)
        for _ in 1:picard_maxit - 1
            gx = EvalF!(GFix!, gx, sol, pdata)
            (picard_beta == 1.0) || (gx = betafix!(gx, sol, picard_beta))
            copy!(res, gx)
            axpy!(-1.0, sol, res)
            resnorm = norm(res)
            updateHist!(ItData, resnorm)
            if verbose
                _aa_verbose_line("Picard", k + 1, resnorm, tol)
            end
            ~keepsolhist || (k += 1; @views solhist[:, k+1] .= sol)
            (resnorm <= picard_tol || resnorm >= resnorm_up_bd) && break
            copy!(sol, gx)
        end
        toosoon = (resnorm <= tol)
    end

    # ----------------------------------------------------------------
    #  Anderson acceleration phase
    # ----------------------------------------------------------------
    if ~toosoon
        copy!(sol, gx)
        alpha = zeros(m + 1)
        k += 1
        ~keepsolhist || (@views solhist[:, k+1] .= sol)
        (gx, dg, df, res, resnorm) =
            aa_point!(gx, GFix!, sol, res, dg, df, beta, pdata)
        updateHist!(ItData, resnorm)
        if verbose
            _aa_verbose_line("Anderson", k, resnorm, tol)
        end
    end

    RF     = zeros(m, m)
    RP     = zeros(m, m)
    ThetA  = zeros(m)
    TmPReS = zeros(m)
    # Device-side temporaries for the m-dimensional coefficient vectors.
    # When gx lives on CPU these are plain Arrays; when on GPU they are device
    # arrays, keeping the two large N-dim mul! calls on-device while the tiny
    # m×m LAPACK solve (RA\tres) stays on CPU.
    tres_dev  = similar(gx, m)
    theta_dev = similar(gx, m)

    while ((k < maxit + picard_maxit) &&
           (resnorm > tol)            &&
           ~toosoon                   &&
           (resnorm < resnorm_up_bd))
        if m == 0
            alphanrm = 1.0
            condit   = 1.0
            copy!(sol, gx)
        else
            BuildDG!(DG, m, k + 1, dg)
            (QP, RP) = aa_qr_update!(QP, RP, df, m, k - 1, Qd)
            mk = min(m, k)
            @views QA    = QP[:, 1:mk]
            @views RA    = RP[1:mk, 1:mk]
            @views theta = ThetA[1:mk]
            @views tres  = TmPReS[1:mk]
            mul!(view(tres_dev, 1:mk), QA', res)
            copyto!(tres, view(tres_dev, 1:mk))
            theta   .= RA \ tres
            condit   = cond(RA)
            alphanrm = falpha(alpha, theta, min(m, k))
            copy!(sol, gx)
            copyto!(view(theta_dev, 1:mk), theta)
            @views mul!(sol, DG[:, 1:mk], view(theta_dev, 1:mk), -1.0, 1.0)
        end
        updateStats!(ItData, condit, alphanrm)
        k += 1
        ~keepsolhist || (@views solhist[:, k+1] .= sol)
        (gx, dg, df, res, resnorm) =
            aa_point!(gx, GFix!, sol, res, dg, df, beta, pdata)
        updateHist!(ItData, resnorm)
        if verbose
            _aa_verbose_line("Anderson", k, resnorm, tol)
        end
    end

    (idid, errcode) = AndersonOK(resnorm, tol, k, m, toosoon, resnorm_up_bd)
    if verbose
        if idid
            errcode == -1 ?
                println("Converged: initial iterate already within tolerance.") :
                println("Converged after ", k, " iterations.")
        elseif errcode == -2
            println("Divergence detected at iteration ", k, ": |res| exceeded upper bound.")
        else
            println("No convergence after ", k, " iterations.")
        end
    end
    aaout = CloseIteration(sol, gx, ItData, idid, errcode, keepsolhist, solhist)
    return aaout
end

"""
BuildDG!(DG, m, k, dg)

Keeps the history of the fixed point map differences.
"""
function BuildDG!(DG, m, k, dg)
    if m == 1
        @views copy!(DG[:, 1], dg)
    elseif k > m + 1
        for ic = 1:m-1
            @views copy!(DG[:, ic], DG[:, ic+1])
        end
        @views copy!(DG[:, m], dg)
    else
        @views copy!(DG[:, k-1], dg)
    end
end

"""
aa_point!(gx, gfix, sol, res, dg, df, beta, pdata)

Evaluate the fixed point map at the new point.
Keep the books to get ready to update the coefficient matrix
for the optimisation problem.
"""
function aa_point!(gx, gfix, sol, res, dg, df, beta, pdata)
    copy!(dg, -gx)
    gx = EvalF!(gfix, gx, sol, pdata)
    (beta == 1.0) || (gx = betafix!(gx, sol, beta))
    axpy!(1.0, gx, dg)
    copy!(df, -res)
    copy!(res, gx)
    axpy!(-1.0, sol, res)
    axpy!(1.0, res, df)
    resnorm = norm(res)
    return (gx, dg, df, res, resnorm)
end

"""
betafix!(gx, sol, beta)

Apply the mixing parameter: gx ← (1-beta)*sol + beta*gx.
"""
function betafix!(gx, sol, beta)
    gx = axpby!((1.0 - beta), sol, beta, gx)
    return gx
end

"""
aa_qr_update!(Q, R, vnew, m, k, Qd)

Update the QR factorisation for the Anderson optimisation problem.
"""
function aa_qr_update!(Q, R, vnew, m, k, Qd)
    (n, m) = size(Q)
    aaqr_dim_check(Q, R, vnew, m, k)
    if k == 0
        R[1, 1] = norm(vnew)
        @views Q[:, 1] .= vnew / norm(vnew)
    else
        if k > m - 1
            downdate_aaqr!(Q, R, m, Qd)
        end
        kq = min(k, m - 1)
        update_aaqr!(Q, R, vnew, m, kq)
    end
    return (Q, R)
end

"""
    Orthogonalize!(Qkm, hv, vnew, mode="cgs2")

Orthogonalize `vnew` against the existing orthonormal columns `Qkm` (an `N×k` matrix),
in place, using classical Gram-Schmidt with one reorthogonalization pass (`"cgs2"`) for
numerical stability. On return: `hv[1:k]` holds the accumulated projection coefficients
`Qkm' * vnew_original` (summed across both passes), `hv[k+1]` holds
`norm(vnew_orthogonalized)`, and `vnew` itself has been overwritten with the new
orthonormalized column (unit norm, orthogonal to every column of `Qkm`) — ready to be
stored directly as the next `Q` column by the caller.
"""
function Orthogonalize!(Qkm, hv, vnew, mode="cgs2")
    mode == "cgs2" || error("Orthogonalize!: unsupported mode $mode (only \"cgs2\" is implemented)")
    k = size(Qkm, 2)
    if k > 0
        # First Gram-Schmidt pass
        h1 = Qkm' * vnew
        mul!(vnew, Qkm, h1, -1.0, 1.0)   # vnew .-= Qkm * h1
        # Second pass (reorthogonalization), accumulated for numerical stability
        h2 = Qkm' * vnew
        mul!(vnew, Qkm, h2, -1.0, 1.0)
        hv[1:k] .= h1 .+ h2
    end
    hv[k+1] = norm(vnew)
    vnew ./= hv[k+1]
    return vnew
end

function update_aaqr!(Q, R, vnew, m, k)
    (nq, mq) = size(Q)
    (k > m - 1) && error("Dimension error in Anderson QR")
    @views Qkm = Q[:, 1:k]
    @views hv  = vec(R[1:k+1, k+1])
    Orthogonalize!(Qkm, hv, vnew, "cgs2")
    @views R[1:k+1, k+1] .= hv
    @views Q[:, k+1]      .= vnew
end

function downdate_aaqr!(Q, R, m, Qd)
    (nq, mq) = size(Q)
    (pd, md) = size(Qd)
    (md == m - 1) || @error("dimension error in downdate")
    @views Rp = R[:, 2:m]
    G  = qr!(Rp)
    Rd = Matrix(G.R)
    Qx = Matrix(G.Q)
    @views R[1:m-1, 1:m-1] .= Rd
    @views R[:, m]          .= 0.0
    # Qx is a small CPU Matrix (m×m); send it to the same device as Q so
    # the large N×m matrix multiply stays on-device.
    Qx_dev = similar(Q, size(Qx)...)
    copyto!(Qx_dev, Qx)
    if (pd == nq)
        mul!(Qd, Q, Qx_dev)
        @views Q[:, 1:m-1] .= Qd
    else
        blocksize = pd
        (dlow, dhigh) = blockdim(nq, blocksize)
        blen = length(dlow)
        for il = 1:blen
            asize = dhigh[il] - dlow[il] + 1
            @views QZ   = Qd[1:asize, :]
            @views Qsec = Q[dlow[il]:dhigh[il], :]
            @views mul!(QZ, Qsec, Qx_dev)
            @views Qsec[:, 1:m-1] .= QZ
        end
    end
    @views Q[:, m] .= 0.0
    return (Q, R)
end

function aaqr_dim_check(Q, R, vnew, m, k)
    (mq, nq) = size(Q)
    (mr, nr) = size(R)
    n      = length(vnew)
    dimqok = ((mq == n) && (nq == m))
    dimrok = ((mr == m) && (nr == m))
    dimok  = (dimqok && dimrok)
    dimok || error("array size error in AA update")
end

function blockdim(n, block)
    p   = Int(floor(n / block))
    res = n - p * block
    ilow  = Int64[]
    ihigh = Int64[]
    for jb = 1:p
        lowval = (jb - 1) * block + 1
        push!(ilow, lowval)
        highval = ilow[jb] + block - 1
        push!(ihigh, highval)
    end
    if res > 0
        lowval = p * block + 1
        push!(ilow, lowval)
        push!(ihigh, n)
    end
    return (ilow, ihigh)
end

# ====================================================================
#  Iteration statistics bookkeeping
# ====================================================================

"""
Mutable struct holding per-iteration diagnostics for aasol.
condhist and alphanorm are initialised with a dummy [1.0] sentinel
that is stripped by CollectStats before being returned to the caller.
"""
mutable struct ItStatsA{T<:Real}
    condhist::Array{T,1}
    alphanorm::Array{T,1}
    history::Array{T,1}
end

function ItStatsA(rnorm)
    ItStatsA([1.0], [1.0], [rnorm])
end

"""
CollectStats(ItData)

Strip the dummy sentinel entries from condhist and alphanorm and return
a clean named tuple suitable for the solver output.
"""
function CollectStats(ItData::ItStatsA)
    stats = (condhist = ItData.condhist[2:end], alphanorm = ItData.alphanorm[2:end])
    return stats
end

function updateStats!(ItData::ItStatsA, condhist, alphanorm)
    append!(ItData.condhist, condhist)
    append!(ItData.alphanorm, alphanorm)
end

function updateHist!(ItData::ItStatsA, rnorm)
    append!(ItData.history, rnorm)
end

# ====================================================================
#  Anderson initialisation
# ====================================================================

"""
Anderson_Init(x0, Vstore, m, maxit, beta, keepsolhist)

Partition Vstore into named working arrays and allocate solution
history if requested. Emits a @warn in low-storage mode (fewer than
3m+3 columns) but is otherwise silent.
"""
function Anderson_Init(x0, Vstore, m, maxit, beta, keepsolhist)
    blocksize = 1024
    (0.0 < abs(beta) <= 1) || error("abs(beta) must be in (0,1]")
    sol = copy(x0)
    n   = length(x0)
    (mv, nv) = size(Vstore)
    mv == n || error("Vstore needs $n rows")
    (nv >= 2 * (m + 1)) || error("Vstore needs $(2*m+4) columns")
    #
    # Reinitialise in case Vstore is being reused across calls.
    #
    Vstore .= 0.0
    if m == 0
        Qd      = []
        QP      = []
        DG      = []
        nvblock = 1
    else
        QP = @views Vstore[:, 1:m]
        DG = @views Vstore[:, m+1:2*m]
        if (nv >= 3 * m + 3)
            Qd      = @views Vstore[:, 2*m+1:3*m-1]
            nvblock = 3 * m
        else
            # Low-storage mode: allocate Qd separately on the heap and warn once.
            @warn "Low storage mode: allocating Qd separately ($(blocksize)×$(m-1)). " *
                  "For best performance allocate Vstore with at least $(3*m+3) columns."
            Qd      = similar(x0, blocksize, m - 1)
            nvblock = 2 * m + 1
        end
    end
    gx      = Anderson_vector_Init(Vstore, nvblock)
    df      = Anderson_vector_Init(Vstore, nvblock + 1)
    dg      = Anderson_vector_Init(Vstore, nvblock + 2)
    res     = Anderson_vector_Init(Vstore, nvblock + 3)
    keepsolhist ? (solhist = solhistinit(n, maxit, sol)) : (solhist = [])
    return (sol, gx, df, dg, res, DG, QP, Qd, solhist)
end

function Anderson_vector_Init(Vstore, nvblock)
    return @views Vstore[:, nvblock]
end

# ====================================================================
#  Exit status and output bundling
# ====================================================================

"""
AndersonOK(resnorm, tol, k, m, toosoon, resnorm_up_bd)

Determine exit status. Returns (idid, errcode) with no side effects —
no messages or warnings are emitted regardless of outcome. Callers
should inspect the tuple (and set verbose=true in aasol if they want
a human-readable summary printed automatically).

errcode meanings:
  -1  initial iterate already satisfied tolerance (toosoon)
  -2  divergence detected (resnorm >= resnorm_up_bd)
   0  converged successfully
  10  maxit reached without convergence
"""
function AndersonOK(resnorm, tol, k, m, toosoon, resnorm_up_bd)
    if toosoon
        return (true, -1)
    elseif resnorm >= resnorm_up_bd
        return (false, -2)
    elseif resnorm <= tol
        return (true, 0)
    else
        return (false, 10)
    end
end

"""
CloseIteration(sol, gx, ItData, idid, errcode, keepsolhist, solhist)

Bundle the solver output into a named tuple. Uses CollectStats to strip
the dummy sentinel from condhist/alphanorm. Pure function: no messages
or warnings are emitted regardless of idid or errcode.
"""
function CloseIteration(sol, gx, ItData, idid, errcode, keepsolhist, solhist)
    stats = CollectStats(ItData)
    return (
        solution    = sol,
        functionval = gx,
        history     = ItData.history,
        stats       = stats,
        idid        = idid,
        errcode     = errcode,
        solhist     = keepsolhist ? solhist : nothing,
    )
end

# ====================================================================
#  falpha — map theta coefficients to alpha and return their l1 norm
# ====================================================================

"""
falpha(alpha, theta, mk)

Map the least-squares solution theta into the convex-combination
coefficients alpha and return their l1 norm (used for diagnostics).
"""
function falpha(alpha, theta, mk)
    alpha[1] = theta[1]
    for ia = 2:mk
        alpha[ia] = theta[ia] - theta[ia-1]
    end
    alpha[mk+1] = 1.0 - theta[mk]
    return norm(alpha, 1)
end

function EvalF!(F!, FS, x, q::Nothing)
    FS = F!(FS, x)
    return FS
end

function EvalF!(F!, FS, x, pdata)
    FS = F!(FS, x, pdata)
    return FS
end

function EvalF!(F!, FS::Real, x::Real, q::Nothing)
    FS = F!(x)
    return FS
end

function EvalF!(F!, FS::Real, x::Real, pdata)
    FS = F!(x, pdata)
    return FS
end