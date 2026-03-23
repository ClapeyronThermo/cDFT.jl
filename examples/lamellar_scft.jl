using cDFT
using DelimitedFiles

# Parameters: symmetric AB diblock above ODT (χN = 30)
N_seg = 20
chi_val = 1.5  # χN = 30
nspecies = 2
rho0 = 1.0
kappa = 25.0

chi = zeros(nspecies, nspecies)
chi[1, 2] = chi_val
chi[2, 1] = chi_val
fh = cDFT.FloryHuggins(chi, rho0, kappa)

L = 7.0
ngrid = 65
n_chains = L / N_seg

seg_spec = vcat(fill(1, N_seg÷2), fill(2, N_seg÷2))
chain = cDFT.SCFTChain(N=N_seg, b=1.0, segment_species=seg_spec,
                        ensemble=:canonical, n_chains=n_chains)

system = cDFT.SCFTSystem(
    interaction=fh,
    chains=[chain],
    nspecies=nspecies,
    species_names=[:A, :B],
    bounds=[0.0, L],
    ngrid=ngrid,
    options=cDFT.DFTOptions()
)

bulk = cDFT.compute_bulk_densities(system)
println("Bulk densities: A=$(bulk[1]), B=$(bulk[2])")

# Initialize with sinusoidal seed
ρ = cDFT.initialize_profiles(system; mode=:uniform)
z = range(0, L, length=ngrid)
amp = 0.1 * bulk[1]
for i in 1:ngrid
    ρ[i, 1] += amp * cos(2π * z[i] / L)
    ρ[i, 2] -= amp * cos(2π * z[i] / L)
end
clamp!(ρ, 1e-10, Inf)

println("Running field-based SCFT convergence...")
result = cDFT.converge_fields!(system, ρ)

# Compute free energy
w = similar(ρ)
cDFT.compute_fields!(system, ρ, w)
dz = cDFT.structure_dz(system.structure)
w_bulk = cDFT.compute_bulk_fields(system.interaction, bulk)
q_fwd, q_bwd, buf, P, iP = cDFT.preallocate_propagator(system, system.propagator, ρ, system.options.device)
cDFT.propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
Q_chains, Q_solvents = cDFT.compute_partition_functions(system, w, w_bulk, q_fwd, dz)
H = cDFT.free_energy(system, ρ, w, Q_chains, Q_solvents)

println("Free energy H = $H")
println("ρ_A range: [$(minimum(ρ[:,1])), $(maximum(ρ[:,1]))]")
println("ρ_B range: [$(minimum(ρ[:,2])), $(maximum(ρ[:,2]))]")
println("ρ_total range: [$(minimum(ρ[:,1].+ρ[:,2])), $(maximum(ρ[:,1].+ρ[:,2]))]")

# Write output
outfile = joinpath(@__DIR__, "lamellar_scft.dat")
data = hcat(collect(z), ρ[:, 1], ρ[:, 2], ρ[:, 1] .+ ρ[:, 2])
open(outfile, "w") do io
    println(io, "# Lamellar SCFT density profiles")
    println(io, "# chi_AB = $chi_val, N = $N_seg, chiN = $(chi_val * N_seg), kappa = $kappa, rho0 = $rho0")
    println(io, "# L = $L, ngrid = $ngrid")
    println(io, "# Free energy H = $H")
    println(io, "# z  rho_A  rho_B  rho_total")
    writedlm(io, data, '\t')
end
println("Wrote density profiles to $outfile")
