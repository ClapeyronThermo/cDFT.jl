"""
    diblock_solution_3d_gpu.jl

3D SCFT example: symmetric AB diblock copolymer in a selective solvent,
computed on a CUDA GPU.

System:
  - 3 species: A (block 1), B (block 2), S (solvent)
  - Symmetric diblock: N=20, f_A = f_B = 0.5, grand canonical
  - Grand canonical solvent with bulk density φ_S = 0.30
  - Total bulk density = 1.0 (quasi-incompressible with κ = 20)
  - χ_AB = 1.5 → χN = 30, well above lamellar ODT (χN ≈ 10.5)
  - χ_AS = 1.0 (solvent avoids A domains)
  - χ_BS = 0.5 (solvent weakly compatible with B domains)

GPU requirements:
  - Install CUDA.jl:   using Pkg; Pkg.add("CUDA")
  - Needs a CUDA-capable GPU and a matching driver

Outputs:
  - diblock_solution_3d_gpu.dat       : ix iy iz x y z ρ_A ρ_B ρ_S ρ_total
  - diblock_solution_3d_gpu_conv.dat  : iter err free_energy  (integer iter)

Run:  julia --project diblock_solution_3d_gpu.jl
Plot: python plot_diblock_3d_gpu.py
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

# ─── Convergence logger ───────────────────────────────────────────────────────
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
N_A      = 10;  N_B = 10   # symmetric diblock
nspecies = 3               # A=1, B=2, S=3
rho0     = 1.0
kappa    = 20.0
b        = 1.0

# Interaction matrix (3×3, symmetric, zero diagonal)
chi_AB = 1.0    # χN = 30
chi_AS = 1.7    # A-solvent repulsion
chi_BS = 0.3    # B-solvent (smaller → solvent prefers B domains)
chi    = [0.0    chi_AB  chi_AS;
          chi_AB 0.0     chi_BS;
          chi_AS chi_BS  0.0   ]
fh     = cDFT.FloryHuggins(chi, rho0, kappa)

# Grand canonical polymer: total segment density = 0.70
# Grand canonical solvent: density = 0.30   (0.70 + 0.30 = ρ₀ = 1.0)
phi_polymer = 0.30
phi_solvent = 0.70

# Box: one lamellar period (L ≈ 7, matching 1D reference)
L     = 30
ngrid = 101     # 33³ grid — runs in minutes on a modern GPU

n_polymer = phi_polymer * L^3 / N_seg
n_polymer = round(Int, n_polymer)  # integer number of polymer chains
n_solvent = L^3 - n_polymer * N_seg
n_solvent = round(Int, n_solvent)  # integer number of solvent molecules
L = (n_polymer * N_seg + n_solvent)^(1/3)  # adjust box size to match integer solvent

# Convert n_polymer and n_solvent to floats
n_polymer = float(n_polymer)
n_solvent = float(n_solvent)

println("=== 3D Diblock Copolymer Solution SCFT (CUDA GPU) ===")
println("N=$N_seg, f_A=0.5, χ_AB=$chi_AB (χN=$(chi_AB*N_seg)), χ_AS=$chi_AS, χ_BS=$chi_BS")
println("φ_polymer=$phi_polymer, φ_solvent=$phi_solvent, κ=$kappa, ρ₀=$rho0")
println("Box: $(L)³,  grid: $(ngrid)³ = $(ngrid^3) points")

# ─── Chain and solvent ────────────────────────────────────────────────────────
seg_spec = vcat(fill(1, N_A), fill(2, N_B))
# chain    = cDFT.SCFTChain(N=N_seg, b=b, segment_species=seg_spec,
#                            ensemble=:grand_canonical,
#                            bulk_density=phi_polymer)
chain      = cDFT.SCFTChain(N=N_seg, b=b, segment_species=seg_spec,
                             ensemble=:canonical, n_chains=n_polymer)

# solvent  = cDFT.SCFTSolvent(species_index=3,
#                              ensemble=:grand_canonical,
#                              bulk_density=phi_solvent)
solvent    = cDFT.SCFTSolvent(species_index=3,
                             ensemble=:canonical, n_molecules=n_solvent)

# ─── System (3D, GPU) ─────────────────────────────────────────────────────────
bounds_3d = [0.0 L; 0.0 L; 0.0 L]
system    = cDFT.SCFTSystem(
    interaction   = fh,
    chains        = [chain],
    solvents      = [solvent],
    nspecies      = nspecies,
    species_names = [:A, :B, :S],
    bounds        = bounds_3d,
    ngrid         = ngrid,
    options       = options,
)

bulk = cDFT.compute_bulk_densities(system)
println("Bulk densities: A=$(round(bulk[1]; sigdigits=5)), " *
        "B=$(round(bulk[2]; sigdigits=5)), S=$(round(bulk[3]; sigdigits=5))")
println("ρ_bulk_total = $(round(sum(bulk); sigdigits=6))")
@assert cDFT.dimension(system) == 3

xgrid = range(0.0, L; length=ngrid)
ygrid = range(0.0, L; length=ngrid)
zgrid = range(0.0, L; length=ngrid)


# ─── Initialize with random perturbation ─────────────────────────────────────
# Random noise allows the system to find the lamellar phase spontaneously.
# The perturbed initializer adds noise on the CPU and adapts to the GPU,
# then normalizes to enforce ρ_total = ρ₀ at every grid point.
ρ = cDFT.initialize_profiles(system; mode=:perturbed, perturbation=0.05)

function write_density_callback(iter, rho, w)
    open(joinpath(@__DIR__, "diblock_solution_3d_gpu_snapshot.dat"), "w") do io
        println(io, "# iter = $(iter)")
        println(io, "# Columns: ix  iy  iz  x  y  z  rho_A  rho_B  rho_S  rho_total")
        for iz in 1:ngrid, iy in 1:ngrid, ix in 1:ngrid
            @printf(io, "%d\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                    ix, iy, iz,
                    xgrid[ix], ygrid[iy], zgrid[iz],
                    rho[ix,iy,iz,1], rho[ix,iy,iz,2], rho[ix,iy,iz,3], sum(rho[ix,iy,iz,:]))
        end
    end
end

# ─── Converge on GPU ─────────────────────────────────────────────────────────
println("\nRunning SCFT Picard iterations on GPU...")
conv_log = ConvergenceCapture()

with_logger(conv_log) do
    global result = cDFT.converge_fields!(system, ρ;
        maxit        = 25000,
        beta         = 0.005,
        tol          = 1e-6,
        anderson_start = 1e-2,
        anderson_m     = 5,
        log_interval = 100,
        save_interval = 100,
        save_callback = write_density_callback,
        verbose      = true,
    )
end

println("\nResult: converged=$(result.converged), iter=$(result.iter), err=$(result.error)")

# ─── Post-process: copy from GPU to CPU ──────────────────────────────────────
ρ_cpu   = Array(ρ)
ρ_A     = ρ_cpu[:, :, :, 1]
ρ_B     = ρ_cpu[:, :, :, 2]
ρ_S     = ρ_cpu[:, :, :, 3]
ρ_total = ρ_A .+ ρ_B .+ ρ_S

println("ρ_A ∈ [$(round(minimum(ρ_A); sigdigits=4)), $(round(maximum(ρ_A); sigdigits=4))]")
println("ρ_B ∈ [$(round(minimum(ρ_B); sigdigits=4)), $(round(maximum(ρ_B); sigdigits=4))]")
println("ρ_S ∈ [$(round(minimum(ρ_S); sigdigits=4)), $(round(maximum(ρ_S); sigdigits=4))]")
println("ρ_total ∈ [$(round(minimum(ρ_total); sigdigits=5)), $(round(maximum(ρ_total); sigdigits=5))]")

amp_A = maximum(ρ_A) - minimum(ρ_A)
amp_B = maximum(ρ_B) - minimum(ρ_B)
println("Lamellar amplitude: A=$(round(amp_A; sigdigits=4)), B=$(round(amp_B; sigdigits=4))")

# ─── Write density output ─────────────────────────────────────────────────────
xgrid = range(0.0, L; length=ngrid)
ygrid = range(0.0, L; length=ngrid)
zgrid = range(0.0, L; length=ngrid)

outfile = joinpath(@__DIR__, "diblock_solution_3d_gpu.dat")
open(outfile, "w") do io
    println(io, "# 3D Diblock Copolymer Solution SCFT (CUDA GPU)")
    println(io, "# N=$(N_seg), f_A=0.5, chi_AB=$(chi_AB), chiN=$(chi_AB*N_seg), chi_AS=$(chi_AS), chi_BS=$(chi_BS)")
    println(io, "# kappa=$(kappa), rho0=$(rho0), phi_polymer=$(phi_polymer), phi_solvent=$(phi_solvent)")
    println(io, "# L=$(L), ngrid=$(ngrid), device=$(CUDA.name(CUDA.device()))")
    println(io, "# converged=$(result.converged), iter=$(result.iter), error=$(result.error)")
    println(io, "# Columns: ix  iy  iz  x  y  z  rho_A  rho_B  rho_S  rho_total")
    for iz in 1:ngrid, iy in 1:ngrid, ix in 1:ngrid
        @printf(io, "%d\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                ix, iy, iz,
                xgrid[ix], ygrid[iy], zgrid[iz],
                ρ_A[ix,iy,iz], ρ_B[ix,iy,iz], ρ_S[ix,iy,iz], ρ_total[ix,iy,iz])
    end
end
println("Wrote density profiles → $outfile")

# ─── Write convergence log (integer iteration column) ────────────────────────
conv_file = joinpath(@__DIR__, "diblock_solution_3d_gpu_conv.dat")
if !isempty(conv_log.iters)
    open(conv_file, "w") do io
        println(io, "# SCFT convergence log (3D GPU)")
        println(io, "# Columns: iter  err  free_energy")
        for (it, err, fe) in zip(conv_log.iters, conv_log.errors, conv_log.free_energies)
            @printf(io, "%d\t%.8e\t%.10f\n", it, err, fe)
        end
    end
    println("Wrote convergence log  → $conv_file")
end
