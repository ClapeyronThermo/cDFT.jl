"""
    symmetric_diblock_1d.jl

1D SCFT example: symmetric AB diblock copolymer forming a lamellar phase.

System:
  - Symmetric diblock: N=20, f_A = f_B = 0.5
  - χN = 30 (well above ODT at χN ≈ 10.5) → lamellar microphase separation
  - Canonical ensemble: fixed number of chains in the box
  - Box length chosen to fit one lamellar period

Outputs:
  - symmetric_1d.dat         : z, ρ_A, ρ_B, ρ_total
  - symmetric_1d_conv.dat    : iter, err, free_energy  (convergence history)

Run: julia --project symmetric_diblock_1d.jl
Plot: python plot_symmetric_1d.py
"""

using cDFT
using DelimitedFiles
using Logging
using Printf

# ─── Capture convergence data from @info log messages ────────────────────────
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
    # Periodic log: "SCFT iter   100 | err = 1.23e-3 | F = 1.23456"
    m = match(r"SCFT iter\s+(\d+)\s*\|\s*err\s*=\s*(\S+)\s*\|\s*F\s*=\s*(\S+)", msg)
    if m !== nothing
        push!(lg.iters, parse(Int, m[1]))
        push!(lg.errors, parse(Float64, m[2]))
        push!(lg.free_energies, parse(Float64, m[3]))
    end
    # Final convergence: "SCFT converged at iter 3735: err = 9.99e-7 | F = 1.6267"
    m2 = match(r"SCFT converged at iter\s+(\d+):\s*err\s*=\s*(\S+)\s*\|\s*F\s*=\s*(\S+)", msg)
    if m2 !== nothing
        push!(lg.iters, parse(Int, m2[1]))
        push!(lg.errors, parse(Float64, m2[2]))
        push!(lg.free_energies, parse(Float64, m2[3]))
    end
    # Print to console
    printstyled("[ Info] "; color=:cyan)
    println(msg)
end

# ─── Parameters ──────────────────────────────────────────────────────────────
N_seg   = 30
chi_val = 1.7      # χN = 30, well above mean-field ODT (χN ≈ 10.5)
f_A     = 0.666667      # symmetric: equal A and B blocks
nspecies = 2
rho0    = 1.0
kappa   = 20.0     # Helfand compressibility
b       = 1.0      # statistical segment length

# Lamellar period: d ≈ 3.8 Rg, Rg = b√(N/6) ≈ 1.83 → d ≈ 7.0
L     = 30
ngrid = 1501        # odd for Simpson quadrature

# Number of chains: set so total segment density = rho0
# For canonical: bulk[α] = n_chains * N_α / V → n_chains = rho0 * V / N
n_chains = rho0 * L / N_seg

# Adjust L with new n_chains
L = n_chains * N_seg / rho0

println("=== 1D Symmetric Diblock Copolymer SCFT ===")
println("N=$N_seg, f_A=$f_A, χ=$chi_val, χN=$(chi_val*N_seg), κ=$kappa")
println("L=$L, ngrid=$ngrid, n_chains=$(round(n_chains; sigdigits=4))")

# ─── Interaction ─────────────────────────────────────────────────────────────
chi = [0.0 chi_val; chi_val 0.0]
fh  = cDFT.FloryHuggins(chi, rho0, kappa)

# ─── Chain definition ────────────────────────────────────────────────────────
N_A    = round(Int, f_A * N_seg)
N_B    = N_seg - N_A
seg_spec = vcat(fill(1, div(N_A, 2)), fill(2, N_B), fill(1, N_A - div(N_A, 2)))   # [1,1,...,2,2,...]
println("Segment specification: $seg_spec")
chain  = cDFT.SCFTChain(N=N_seg, b=b, segment_species=seg_spec,
                         ensemble=:canonical, n_chains=n_chains)

# ─── System ──────────────────────────────────────────────────────────────────
system = cDFT.SCFTSystem(
    interaction   = fh,
    chains        = [chain],
    nspecies      = nspecies,
    species_names = [:A, :B],
    bounds        = [0.0, L],
    ngrid         = ngrid,
    options       = cDFT.DFTOptions()
)

bulk = cDFT.compute_bulk_densities(system)
println("Bulk densities: A=$(round(bulk[1]; sigdigits=6)), B=$(round(bulk[2]; sigdigits=6))")
println("Rg = $(round(b*sqrt(N_seg/6); sigdigits=3))")

# ─── Initialize with cosine seed to break lamellar symmetry ──────────────────
ρ = cDFT.initialize_profiles(system; mode=:uniform)
z = range(0.0, L; length=ngrid)
amp = 0.1 * bulk[1]
for i in 1:ngrid
    ρ[i, 1] += amp * cos(2π * z[i] / (L/6))
    ρ[i, 2] -= amp * cos(2π * z[i] / (L/6))
end
clamp!(ρ, 1e-10, Inf)

# ─── Run SCFT convergence (capture convergence log) ──────────────────────────
println("\nRunning SCFT Picard iterations...")
conv_log = ConvergenceCapture()

function snapshot_callback(iter, rho, w)
    open("symmetric_1d_snapshot.dat", "w") do io
        println(io, "# iter = $(iter)")
        println(io, "# z\trho_A\trho_B\tw_A\tw_B")
        for i in axes(rho, 1)
            @printf(io, "%.6f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                z[i], rho[i,1], rho[i,2], w[i,1], w[i,2])
        end
    end
end

with_logger(conv_log) do
    global result = cDFT.converge_fields!(system, ρ;
        maxit          = 20000,
        beta           = 0.0025,
        tol            = 1e-6,
        anderson_start = 1e-2,   # switch from Picard to Anderson when err < 1e-2
        anderson_m     = 5,      # number of Anderson history vectors
        log_interval   = 100,
        save_interval  = 500,
        save_callback  = snapshot_callback,
        verbose        = true,
    )
end

println("\nResult: converged=$(result.converged), iter=$(result.iter), err=$(result.error)")

# ─── Post-process ────────────────────────────────────────────────────────────
ρ_A     = ρ[:, 1]
ρ_B     = ρ[:, 2]
ρ_total = ρ_A .+ ρ_B

amp_A = maximum(ρ_A) - minimum(ρ_A)
amp_B = maximum(ρ_B) - minimum(ρ_B)
println("ρ_A amplitude: $(round(amp_A; sigdigits=4))")
println("ρ_B amplitude: $(round(amp_B; sigdigits=4))")
println("ρ_total ∈ [$(round(minimum(ρ_total); sigdigits=5)), $(round(maximum(ρ_total); sigdigits=5))]")

# Free energy at convergence
w = similar(ρ)
cDFT.compute_fields!(system, ρ, w)
dz_val  = cDFT.structure_dz(system.structure)
w_bulk  = cDFT.compute_bulk_fields(system.interaction, bulk)
q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, system.propagator, ρ, system.options.device)
cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz_val)
H = cDFT.free_energy(system, ρ, w, Q_chains, Q_solvents)
println("Free energy H = $(round(H; sigdigits=8))")

# ─── Write density profiles ───────────────────────────────────────────────────
outfile = joinpath(@__DIR__, "symmetric_1d.dat")
header  = """# 1D Symmetric Diblock Copolymer SCFT - Lamellar Phase
# N=$(N_seg), f_A=$(f_A), chi=$(chi_val), chiN=$(chi_val*N_seg), kappa=$(kappa), rho0=$(rho0)
# L=$(L), ngrid=$(ngrid), n_chains=$(round(n_chains; sigdigits=6))
# converged=$(result.converged), iter=$(result.iter), error=$(result.error)
# Free energy H=$(H)
# Columns: z  rho_A  rho_B  rho_total"""
data    = hcat(collect(z), ρ_A, ρ_B, ρ_total)
open(outfile, "w") do io
    println(io, header)
    writedlm(io, data, '\t')
end
println("Wrote density profiles → $outfile")

# ─── Write convergence history ────────────────────────────────────────────────
conv_file = joinpath(@__DIR__, "symmetric_1d_conv.dat")
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
