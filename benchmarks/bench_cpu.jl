"""
    bench_cpu.jl

Publication-quality benchmark of SCFT iteration performance on CPU (Float64).

Run:
    julia -t 4 --project benchmarks/bench_cpu.jl <hardware_tag>

Example:
    julia -t 4 --project benchmarks/bench_cpu.jl M4-Max
    julia -t 4 --project benchmarks/bench_cpu.jl i9-10850K

The hardware tag is embedded in the output CSV filenames so results from
different machines can coexist without overwriting each other.

What is benchmarked:
    - Polymeric system: symmetric AB diblock copolymer (canonical ensemble)
    - Monomeric system: single-species solvent (grand-canonical ensemble)

For each system type, sweeps 1D, 2D, and 3D at several grid sizes.
Reports the median time per SCFT iteration step, excluding JIT startup.
Each data point averages over N_STEPS=8 consecutive Picard steps
(Anderson acceleration disabled so cost is purely the field-propagation loop).

Outputs:
    bench_cpu_<hardware_tag>_polymer_results.csv
    bench_cpu_<hardware_tag>_solvent_results.csv
"""

using cDFT
using BenchmarkTools
using Statistics
using Printf

if isempty(ARGS)
    error("Usage: julia -t 4 --project benchmarks/bench_cpu.jl <hardware_tag>\n" *
          "Example: julia -t 4 --project benchmarks/bench_cpu.jl M4-Max")
end
const HW_TAG = ARGS[1]

BenchmarkTools.DEFAULT_PARAMETERS.samples  = 5
BenchmarkTools.DEFAULT_PARAMETERS.evals    = 1

# ── Benchmark settings ────────────────────────────────────────────────────────
const N_STEPS = 100       # iterations averaged per timing sample

const SIZES_1D = [64, 128, 256, 512, 1024, 2048, 4096]
const SIZES_2D = [32, 64, 128, 256, 512]
const SIZES_3D = [16, 32, 64, 128]

# ── Polymer (AB diblock) parameters ──────────────────────────────────────────
const N_seg   = 30
const N_A     = 15
const N_B     = N_seg - N_A
const chi_val = 1.0
const rho0    = 1.0
const kappa   = 20.0
const b_stat  = 1.0
const L       = 10.0     # box length per direction (fixed; only grid density varies)

# ── System builders ───────────────────────────────────────────────────────────

"""
Build bounds and ngrid for the given dimension and grid size.
Returns (ngrid, bounds) where:
  1D → ngrid=Int,       bounds=Vector (length 2)
  2D → ngrid=(ng,ng),   bounds=2×2 Matrix
  3D → ngrid=Int,       bounds=3×2 Matrix
"""
function grid_args(nd::Int, ng::Int)
    if nd == 1
        return ng, [0.0, L]
    elseif nd == 2
        return (ng, ng), [0.0 L; 0.0 L]
    else
        return ng, [0.0 L; 0.0 L; 0.0 L]
    end
end

function make_polymer_system(nd::Int, ng::Int, options)
    ngrid, bounds = grid_args(nd, ng)
    V        = L^nd
    n_chains = rho0 * V / N_seg

    chi    = [0.0 chi_val; chi_val 0.0]
    fh     = cDFT.FloryHuggins(chi, rho0, kappa)

    seg_spec = vcat(fill(1, N_A), fill(2, N_B))
    chain    = cDFT.SCFTChain(N=N_seg, b=b_stat, segment_species=seg_spec,
                               ensemble=:canonical, n_chains=n_chains)

    return cDFT.SCFTSystem(
        interaction   = fh,
        chains        = [chain],
        nspecies      = 2,
        species_names = [:A, :B],
        bounds        = bounds,
        ngrid         = ngrid,
        options       = options,
    )
end

function make_solvent_system(nd::Int, ng::Int, options)
    ngrid, bounds = grid_args(nd, ng)

    # 1×1 χ matrix (self-interaction is zero for single species)
    chi_1x1 = reshape([0.0], 1, 1)
    fh      = cDFT.FloryHuggins(chi_1x1, rho0, kappa)

    # Grand-canonical solvent: density computed directly from field via
    # ρ_S(r) = ρ_S^bulk · exp(w_bulk - w(r)), no chain propagation needed.
    solvent = cDFT.SCFTSolvent(
        species_index = 1,
        ensemble      = :grand_canonical,
        bulk_density  = rho0,
    )

    return cDFT.SCFTSystem(
        interaction   = fh,
        chains        = cDFT.SCFTChain[],
        solvents      = [solvent],
        nspecies      = 1,
        species_names = [:S],
        bounds        = bounds,
        ngrid         = ngrid,
        options       = options,
    )
end

# ── Core benchmark function ───────────────────────────────────────────────────

"""
    run_bench(system, ρ_init) → BenchmarkTools.Trial

Times N_STEPS consecutive Picard SCFT steps, excluding JIT startup.
Anderson acceleration is disabled (anderson_start=0) and convergence
tolerance is set to 0 (tol=0) so all N_STEPS iterations always execute.
"""
function run_bench(system, ρ_init)
    ρ_bench = copy(ρ_init)

    # Warmup: first call triggers JIT compilation — excluded from timing.
    cDFT.converge_fields!(system, copy(ρ_init);
        maxit          = N_STEPS,
        tol            = 0.0,        # never converge early
        anderson_start = 0.0,        # pure Picard throughout (no AA overhead)
        log_interval   = 0,
        verbose        = false,
    )

    b = @benchmark begin
        $ρ_bench .= $ρ_init
        cDFT.converge_fields!($system, $ρ_bench;
            maxit          = $N_STEPS,
            tol            = 0.0,
            anderson_start = 0.0,
            log_interval   = 0,
            verbose        = false,
        )
    end samples=5 evals=1

    return b
end

# ── CSV writer ────────────────────────────────────────────────────────────────

function write_csv(filename, header_lines, rows)
    open(filename, "w") do io
        for h in header_lines
            println(io, "# $h")
        end
        println(io, "# Columns: dim,ng_per_side,n_grid_pts,t_median_ms,t_std_ms")
        for (dim, ng, npts, t_med, t_std) in rows
            @printf(io, "%s,%d,%d,%.6f,%.6f\n", dim, ng, npts, t_med, t_std)
        end
    end
    println("  → Wrote $filename")
end

# ── Main sweep ────────────────────────────────────────────────────────────────

function run_all()
    options = cDFT.DFTOptions()    # CPU, Float64

    cpu_info = Sys.cpu_info()[1].model
    n_threads = Threads.nthreads()

    println("=" ^ 72)
    println("SCFT CPU Benchmark  |  Float64  |  Threads: $n_threads")
    println("CPU: $cpu_info")
    println("Polymer: N_seg=$N_seg, N_A=$N_A, χ=$chi_val, κ=$kappa, ρ₀=$rho0, b=$b_stat, L=$L")
    println("N_STEPS=$N_STEPS iterations per sample, 5 samples per grid size")
    println("=" ^ 72)

    global_header = [
        "Backend: CPU  Float64  Threads: $n_threads",
        "CPU: $cpu_info",
        "N_seg=$N_seg  N_A=$N_A  chi=$chi_val  kappa=$kappa  rho0=$rho0  b=$b_stat  L=$L",
        "N_STEPS=$N_STEPS  samples=5  evals=1  anderson_start=0  tol=0",
    ]

    systems_to_bench = [
        ("Polymer (AB diblock copolymer)", make_polymer_system,
         "bench_cpu_$(HW_TAG)_polymer_results.csv",
         "System: diblock copolymer  N=$N_seg N_A=$N_A chi=$chi_val kappa=$kappa rho0=$rho0"),
        ("Solvent (monomeric, grand-canonical)", make_solvent_system,
         "bench_cpu_$(HW_TAG)_solvent_results.csv",
         "System: monomeric solvent  kappa=$kappa rho0=$rho0"),
    ]

    for (sys_label, make_sys, outfile, sys_header) in systems_to_bench
        println("\n── $sys_label " * "─"^(72 - length(sys_label) - 4))
        @printf("  %-4s  %-8s  %-14s  %s\n", "Dim", "ng/side", "N_grid_pts", "t_median [ms/step]")
        println("  " * "─"^58)

        rows = Tuple{String,Int,Int,Float64,Float64}[]

        for (dimlabel, nd, sizes) in [("1D", 1, SIZES_1D), ("2D", 2, SIZES_2D), ("3D", 3, SIZES_3D)]
            for ng in sizes
                n_pts = ng^nd

                sys = make_sys(nd, ng, options)
                ρ0  = cDFT.initialize_profiles(sys; mode=:uniform)

                b = run_bench(sys, ρ0)

                t_med = median(b.times) / 1e6 / N_STEPS   # ns → ms, per step
                t_std = std(b.times)   / 1e6 / N_STEPS

                @printf("  %-4s  %-8d  %-14d  %.4f ± %.4f ms\n",
                        dimlabel, ng, n_pts, t_med, t_std)

                push!(rows, (dimlabel, ng, n_pts, t_med, t_std))
            end
            println()
        end

        write_csv(outfile, [global_header..., sys_header], rows)
    end

    println("\nBenchmark complete.")
end

run_all()
