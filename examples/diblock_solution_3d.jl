using cDFT
using DelimitedFiles

# ── Parameters ──────────────────────────────────────────────────────────
# Using parameters proven to work in 1D lamellar test
N_A = 10
N_B = 10
N_seg = N_A + N_B
nspecies = 2  # A=1, B=2

chi_AB = 1.5   # χN = 30
rho0 = 1.0
kappa = 25.0
b = 1.0

# Box: small cube matching one lamellar period for fast convergence
Lx = 7.0
Ly = 7.0
Lz = 7.0
dx = 0.5
ngrid_x = round(Int, Lx / dx)
ngrid_y = round(Int, Ly / dx)
ngrid_z = round(Int, Lz / dx)

# ── Interaction ─────────────────────────────────────────────────────────
chi = zeros(nspecies, nspecies)
chi[1, 2] = chi_AB;  chi[2, 1] = chi_AB
fh = cDFT.FloryHuggins(chi, rho0, kappa)

# ── Chain definition (grand canonical melt) ─────────────────────────────
seg_spec = vcat(fill(1, N_A), fill(2, N_B))
chain = cDFT.SCFTChain(N=N_seg, b=b, segment_species=seg_spec,
                        ensemble=:grand_canonical, bulk_density=rho0)

# ── Build system ────────────────────────────────────────────────────────
bounds = [0.0 Lx; 0.0 Ly; 0.0 Lz]
options = cDFT.DFTOptions()

println("Building SCFTSystem: $(ngrid_x)×$(ngrid_y)×$(ngrid_z) grid, $nspecies species...")
system = cDFT.SCFTSystem(
    interaction=fh,
    chains=[chain],
    nspecies=nspecies,
    species_names=[:A, :B],
    bounds=bounds,
    ngrid=ngrid_x,
    options=options
)

bulk = cDFT.compute_bulk_densities(system)
println("Bulk densities: A=$(bulk[1]), B=$(bulk[2])")
println("χN = $(chi_AB * N_seg)")

# ── Initialize with lamellar seed along z-axis ──────────────────────────
println("Initializing profiles with lamellar seed...")
ρ = cDFT.initialize_profiles(system; mode=:uniform)
coords = range(0, Lz, length=ngrid_z)

# Seed lamellar perturbation along z with period ≈ 7 (known good from 1D)
d_lam = 7.0
amp = 0.1 * bulk[1]
for iz in 1:ngrid_z
    δ = amp * cos(2π * coords[iz] / d_lam)
    ρ[:, :, iz, 1] .+= δ
    ρ[:, :, iz, 2] .-= δ
end
clamp!(ρ, 1e-10, Inf)

# ── Converge (5000 iterations max) ──────────────────────────────────────
println("Running field-based SCFT convergence (5000 iterations max)...")
result = cDFT.converge_fields!(system, ρ; picard_maxit=0, anderson_maxit=5000, beta=1e-2)

# ── Post-process ────────────────────────────────────────────────────────
ρ_A = ρ[:, :, :, 1]
ρ_B = ρ[:, :, :, 2]
ρ_total = ρ_A .+ ρ_B

println("ρ_A range: [$(minimum(ρ_A)), $(maximum(ρ_A))]")
println("ρ_B range: [$(minimum(ρ_B)), $(maximum(ρ_B))]")
println("ρ_total range: [$(minimum(ρ_total)), $(maximum(ρ_total))]")
println("ρ_A amplitude: $(maximum(ρ_A) - minimum(ρ_A))")

# ── Write to file ───────────────────────────────────────────────────────
outfile = joinpath(@__DIR__, "diblock_solution_3d.dat")
println("Writing density profiles to $outfile...")

cx = range(0, Lx, length=ngrid_x)
cy = range(0, Ly, length=ngrid_y)
cz = range(0, Lz, length=ngrid_z)
open(outfile, "w") do io
    println(io, "# 3D Diblock Copolymer Melt SCFT Density Profiles")
    println(io, "# chi_AB=$chi_AB, N=$N_seg, chiN=$(chi_AB * N_seg)")
    println(io, "# rho0=$rho0, kappa=$kappa, b=$b")
    println(io, "# Lx=$Lx, Ly=$Ly, Lz=$Lz, ngrid=$ngrid_x, dx=$dx")
    println(io, "# ensemble=grand_canonical, bulk_density=$rho0")
    println(io, "# Columns: ix iy iz x y z rho_A rho_B rho_total")
    for iz in 1:ngrid_z
        for iy in 1:ngrid_y
            for ix in 1:ngrid_x
                x = cx[ix]
                y = cy[iy]
                z = cz[iz]
                println(io, "$ix\t$iy\t$iz\t$x\t$y\t$z\t$(ρ_A[ix,iy,iz])\t$(ρ_B[ix,iy,iz])\t$(ρ_A[ix,iy,iz]+ρ_B[ix,iy,iz])")
            end
        end
    end
end

println("Done. Output written to $outfile")
