"""
    asymmetric_diblock_2d.jl

2D SCFT example: asymmetric AB diblock copolymer forming a hexagonally-packed
cylinder phase.

System:
  - Asymmetric diblock: N=20, N_A=7 (f_A=0.35), N_B=13
  - χN = 35 (above the cylinder ODT, χN ≈ 14.5 at f_A=0.35)
  - Canonical ensemble
  - Box ≈ one hexagonal unit cell: Lx = d, Ly = d·√3/2 where d ≈ 6.5

Phase expected: A-rich cylinder at box center surrounded by B matrix.
The minority A block (f_A=0.35) forms the cylinder core.

Outputs:
  - asymmetric_2d.dat         : ix, iy, x, y, ρ_A, ρ_B, ρ_total
  - asymmetric_2d_conv.dat    : iter, err, free_energy

Run: julia --project asymmetric_diblock_2d.jl
Plot: python plot_asymmetric_2d.py
"""

using cDFT
using CUDA                 # install with: using Pkg; Pkg.add("CUDA")
using DelimitedFiles
using Logging
using Printf

# ─── GPU setup ───────────────────────────────────────────────────────────────
if !CUDA.functional()
    error("CUDA GPU not available.\n" *
          "Check your installation with:  using CUDA; CUDA.versioninfo()")
end

device  = CUDABackend()
options = cDFT.DFTOptions(device)
println("GPU: $(CUDA.name(CUDA.device()))")
println("VRAM available: $(round(CUDA.available_memory()/1024^3; sigdigits=3)) GB")

# ─── Convergence logger (same helper as 1D example) ──────────────────────────
struct ConvergenceCapture <: AbstractLogger
    iters::Vector{Int}
    errors::Vector{Float64}
    free_energies::Vector{Float64}
end
ConvergenceCapture() = ConvergenceCapture(Int[], Float64[], Float64[])

Logging.min_enabled_level(::ConvergenceCapture) = Logging.Info
Logging.shouldlog(::ConvergenceCapture, args...) = true
function Logging.handle_message(lg::ConvergenceCapture, level, message,
                                 _module, group, id, file, line; kwargs...)
    msg = string(message)
    m = match(r"SCFT iter\s+(\d+)\s*\|\s*err\s*=\s*(\S+)\s*\|\s*F\s*=\s*(\S+)", msg)
    if m !== nothing
        push!(lg.iters, parse(Int, m[1]))
        push!(lg.errors, parse(Float64, m[2]))
        push!(lg.free_energies, parse(Float64, m[3]))
    end
    m2 = match(r"SCFT converged at iter\s+(\d+):\s*err\s*=\s*(\S+)\s*\|\s*F\s*=\s*(\S+)", msg)
    if m2 !== nothing
        push!(lg.iters, parse(Int, m2[1]))
        push!(lg.errors, parse(Float64, m2[2]))
        push!(lg.free_energies, parse(Float64, m2[3]))
    end
    printstyled("[ Info] "; color=:cyan)
    println(msg)
end

# ─── Parameters ──────────────────────────────────────────────────────────────
N_seg    = 20
N_A      = 10        # minority A block  (f_A = 0.35)
N_B      = N_seg - N_A
f_A      = N_A / N_seg
chi_val  = 1.7     # χN = 35, above cylinder ODT at f_A=0.35
nspecies = 2
rho0     = 1.0
kappa    = 20.0
b        = 1.0

# Hexagonal cylinder unit cell
# d_hex  ≈ 3.6·Rg, Rg = b·√(N/6) ≈ 1.83
# d_hex  ≈ 6.5;  Ly = d_hex·√3/2 ≈ 5.63
Lx   = 80
Ly   = 80   # ≈ 5.63
ngx  = 320
ngy  = 320                  # ≈ Ly / (Lx/(ngx-1)) for square pixels

# Number of chains (canonical): ρ0 = n_chains · N / V
V_approx = Lx * Ly
n_chains = rho0 * V_approx / N_seg

# recompute Lx and Ly based on n_chains (square box)
Lx = sqrt(n_chains * N_seg / rho0)
Ly = Lx

println("=== 2D Asymmetric Diblock Copolymer SCFT (Cylinder Phase) ===")
println("N=$N_seg, N_A=$N_A, N_B=$N_B, f_A=$(round(f_A; sigdigits=3))")
println("χ=$chi_val, χN=$(chi_val*N_seg), κ=$kappa, ρ₀=$rho0")
println("Box: Lx=$(round(Lx; sigdigits=4)) × Ly=$(round(Ly; sigdigits=4))")
println("Grid: $(ngx) × $(ngy), n_chains=$(round(n_chains; sigdigits=4))")

# ─── Interaction ─────────────────────────────────────────────────────────────
chi = [0.0 chi_val; chi_val 0.0]
fh  = cDFT.FloryHuggins(chi, rho0, kappa)

# ─── Chain definition ────────────────────────────────────────────────────────
seg_spec = vcat(fill(1, N_A), fill(2, N_B))
chain    = cDFT.SCFTChain(N=N_seg, b=b, segment_species=seg_spec,
                           ensemble=:canonical, n_chains=n_chains)

# ─── System (2D) ─────────────────────────────────────────────────────────────
bounds_2d = [0.0 Lx; 0.0 Ly]
system    = cDFT.SCFTSystem(
    interaction   = fh,
    chains        = [chain],
    nspecies      = nspecies,
    species_names = [:A, :B],
    bounds        = bounds_2d,
    ngrid         = (ngx, ngy),
    options       = options
)

bulk = cDFT.compute_bulk_densities(system)
println("Bulk densities: A=$(round(bulk[1]; sigdigits=5)), B=$(round(bulk[2]; sigdigits=5))")
@assert cDFT.dimension(system) == 2

xgrid = range(0.0, Lx; length=ngx)
ygrid = range(0.0, Ly; length=ngy)

# ─── Initialize: random perturbation from bulk ────────────────────────────────
# Random noise breaks all symmetries, letting the system find its preferred
# phase (cylinder) without biasing the orientation or position.
# The initialize_profiles helper adds noise and re-normalizes to enforce
# incompressibility (ρ_total = ρ₀ everywhere at the start).
ρ = cDFT.initialize_profiles(system; mode=:perturbed, perturbation=0.05)

# ─── (Alternative) Gaussian cylinder seed ────────────────────────────────────
# Uncomment to seed with a single A-rich Gaussian spot at the box center.
# This biases the system toward a cylinder at a known position and can
# reduce the number of iterations needed for convergence.
#
# ρ    = cDFT.initialize_profiles(system; mode=:uniform)
# amp   = 0.2 * bulk[1]   # perturbation amplitude
# σ_cyl = 0.9              # cylinder core radius (in units of b)
# for ix in 1:ngx, iy in 1:ngy
#     x = xgrid[ix]; y = ygrid[iy]
#     r2 = (x - Lx/2)^2 + (y - Ly/2)^2
#     δ = amp * exp(-r2 / (2 * σ_cyl^2))
#     ρ[ix, iy, 1] += δ
#     ρ[ix, iy, 2] -= δ
# end
# clamp!(ρ, 1e-10, Inf)
# ρ_total_seed = sum(ρ, dims=3)
# ρ .*= rho0 ./ ρ_total_seed

# ─── Snapshot callback (overwrites file each call) ───────────────────────────
function write_density_callback(iter, rho, w)
    open(joinpath(@__DIR__, "asymmetric_2d_snapshot.dat"), "w") do io
        println(io, "# iter = $(iter)")
        println(io, "# Columns: ix  iy  x  y  rho_A  rho_B")
        for iy in 1:ngy, ix in 1:ngx
            @printf(io, "%d\t%d\t%.6f\t%.6f\t%.8f\t%.8f\n",
                ix, iy, xgrid[ix], ygrid[iy], rho[ix,iy,1], rho[ix,iy,2])
        end
    end
end

# ─── Converge ────────────────────────────────────────────────────────────────
println("\nRunning SCFT Picard iterations...")
conv_log = ConvergenceCapture()

with_logger(conv_log) do
    global result = cDFT.converge_fields!(system, ρ;
        maxit          = 20000,
        beta           = 0.005,
        tol            = 1e-6,
        anderson_start = 1e-2,   # switch from Picard to Anderson when err < 1e-2
        anderson_m     = 5,      # number of Anderson history vectors
        log_interval   = 50,
        save_interval  = 100,
        save_callback  = write_density_callback,
        verbose        = true,
    )
end

println("\nResult: converged=$(result.converged), iter=$(result.iter), err=$(result.error)")

# ─── Post-process ────────────────────────────────────────────────────────────
ρ_A     = ρ[:, :, 1]
ρ_B     = ρ[:, :, 2]
ρ_total = ρ_A .+ ρ_B

println("ρ_A ∈ [$(round(minimum(ρ_A); sigdigits=4)), $(round(maximum(ρ_A); sigdigits=4))]")
println("ρ_B ∈ [$(round(minimum(ρ_B); sigdigits=4)), $(round(maximum(ρ_B); sigdigits=4))]")
println("ρ_total ∈ [$(round(minimum(ρ_total); sigdigits=5)), $(round(maximum(ρ_total); sigdigits=5))]")

# Check cylinder formation: A maximum should be near box center
imax_A = argmax(ρ_A)
println("ρ_A maximum at grid ($(imax_A[1]), $(imax_A[2])) → " *
        "x=$(round(xgrid[imax_A[1]]; sigdigits=3)), y=$(round(ygrid[imax_A[2]]; sigdigits=3))")

# Free energy
w = similar(ρ)
cDFT.compute_fields!(system, ρ, w)
dz_val  = cDFT.structure_dz(system.structure)
w_bulk  = cDFT.compute_bulk_fields(system.interaction, bulk)
q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, system.propagator, ρ, system.options.device)
cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz_val)
H = cDFT.free_energy(system, ρ, w, Q_chains, Q_solvents)
println("Free energy H = $(round(H; sigdigits=8))")

# ─── Write density output ─────────────────────────────────────────────────────
outfile = joinpath(@__DIR__, "asymmetric_2d.dat")
open(outfile, "w") do io
    println(io, "# 2D Asymmetric Diblock Copolymer SCFT - Hexagonal Cylinder Phase")
    println(io, "# N=$(N_seg), N_A=$(N_A), f_A=$(round(f_A;sigdigits=3)), chi=$(chi_val), chiN=$(chi_val*N_seg)")
    println(io, "# kappa=$(kappa), rho0=$(rho0), b=$(b)")
    println(io, "# Lx=$(round(Lx;sigdigits=5)), Ly=$(round(Ly;sigdigits=5)), ngx=$(ngx), ngy=$(ngy)")
    println(io, "# converged=$(result.converged), iter=$(result.iter), error=$(result.error)")
    println(io, "# Free energy H=$(H)")
    println(io, "# Columns: ix  iy  x  y  rho_A  rho_B  rho_total")
    for iy in 1:ngy, ix in 1:ngx
        @printf(io, "%d\t%d\t%.6f\t%.6f\t%.8f\t%.8f\t%.8f\n",
                ix, iy, xgrid[ix], ygrid[iy],
                ρ_A[ix, iy], ρ_B[ix, iy], ρ_total[ix, iy])
    end
end
println("Wrote density profiles → $outfile")

# ─── Write convergence log ────────────────────────────────────────────────────
conv_file = joinpath(@__DIR__, "asymmetric_2d_conv.dat")
if !isempty(conv_log.iters)
    open(conv_file, "w") do io
        println(io, "# SCFT convergence log")
        println(io, "# Columns: iter  err  free_energy")
        for (it, err, fe) in zip(conv_log.iters, conv_log.errors, conv_log.free_energies)
            @printf(io, "%d\t%.8e\t%.10f\n", it, err, fe)
        end
    end
    println("Wrote convergence log  → $conv_file")
end
