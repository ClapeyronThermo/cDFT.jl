# AGENTS.md — cDFT.jl PC-SAFT analysis task

## Mission

You are working inside a Julia repository named `cDFT.jl`. Your goal is to read the repository structure and source code, then write a technical report at:

```text
./benchmark/current.md
```

The report must explain the PC-SAFT cDFT implementation at both the mathematical/theoretical level and the code-execution level. The report must be concrete, file-path aware, and tied to the actual source code in this repository. Do not invent APIs or file names. If an expected symbol or file is missing, state that clearly and show the search you used.

## Required output file

Create or overwrite:

```text
benchmark/current.md
```

The document must contain these four sections:

1. **PC-SAFT cDFT theory and code path**

   - Explain the residual Helmholtz/free-energy functional used by the PC-SAFT density functional implementation.
   - Explain the weighted-density construction.
   - Explain the PC-SAFT local residual free-energy density decomposition, including hard-sphere, hard-chain, dispersion, and association pieces if present in the code.
   - Map each formula to the concrete Julia function(s) and file path(s).
   - For each major function, state input, output, array shape, and physical meaning.

2. **Minimal 1D PC-SAFT benchmark to `benchmark/minima.jl`**

   - Provide a minimal runnable Julia benchmark script for a 1D PC-SAFT DFT system.

   - The grid count must be adjustable by a command-line argument or a variable near the top of the script.

   - The script should construct a PC-SAFT model, define a 1D DFT structure/domain, build a density profile, evaluate the free energy and/or derivative, and then run a minimal minimization if the repository exposes a minimization API.

   - Save this benchmark script as:

     ```text
     benchmark/minima.jl
     ```

   - If the exact construction API is unclear, write a verified partial benchmark and include TODO comments only at the exact API gaps. Do not fabricate missing constructors.

3. **CPU autodiff step**

   - Identify the exact line(s) where automatic differentiation occurs.
   - Explain whether this AD is whole-functional AD or local AD.
   - State the mathematical derivative it computes.
   - State the input array shape and output array shape of the AD call.
   - State whether the AD call runs on CPU threads or GPU kernels, based on the code.
   - Explain how this local derivative is converted into the full functional derivative.

4. **GPU migration plan**

   - Identify which files/functions must be changed to move the current CPU autodiff part to GPU.
   - At minimum inspect the file containing `F_res`, `δFδρ_res!`, and `preallocate`, and the PC-SAFT model file containing `f_res(::PCSAFTModel, ...)`.
   - Explain whether the main changes belong in `model.jl`, `PCSAFT.jl`, or both.
   - Propose a practical implementation plan: hand-coded analytic gradient, Enzyme/K Enzyme GPU kernel, KernelAbstractions kernel, or a hybrid approach.
   - Explicitly list blockers such as scalar Julia functions that are not GPU-compatible, allocations inside `f_res`, use of `ForwardDiff.gradient!`, non-isbits caches, unsupported Clapeyron calls, or CPU-only `Threads.@threads`.

## Repository-inspection protocol

Start by mapping the repository. Run these commands from the repository root and quote relevant results in `benchmark/current.md`:

```bash
pwd
find . -maxdepth 3 -type f | sort
find . -maxdepth 4 -type f | grep -E '(PCSAFT|model|dft|field|minim|preallocate|propagat|external|BasicIdeal|DGT|Project|Manifest)'
grep -R "function F_res\|function δFδρ_res\|ForwardDiff.gradient\|function f_res\|PCSAFTSpecies\|get_fields\|function preallocate\|function minima\|optimize\|minimi" -n .
```

Then open and read the relevant files before writing conclusions. Use exact file paths and function names.

## Known high-level code structure to verify

The repository appears to have a top-level module that imports Clapeyron and includes files similar to:

```julia
import Clapeyron: a_res
include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")
```

A core model file is expected to define functions similar to:

```julia
F_res(system::AbstractcDFTSystem, ρ)
δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool)
δFδρ_res(system::AbstractcDFTSystem, ρ)
length_scales(model::EoSModel)
```

A PC-SAFT file is expected to define functions similar to:

```julia
struct PCSAFTSpecies <: DFTSpecies
get_fields(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
get_species(model::PCSAFTModel, structure::DFTStructure)
get_propagator(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure)
f_res(system::Union{DFTSystem,ElectrolyteDFTSystem}, model::PCSAFTModel, n)
f_hc(...)
f_disp(...)
Δ(...)
length_scale(model::SAFTModel)
```

Do not assume these files are named exactly `model.jl` or `PCSAFT.jl`; verify using `grep` and `find`.

## Mathematical content that must appear in `current.md`

### Functional definition

Write the residual free-energy functional as:

```math
F_{\mathrm{res}}[\rho]
= \int_\Omega f_{\mathrm{res}}(\mathbf n(\mathbf r))\,d\mathbf r.
```

The weighted densities are:

```math
n_{\alpha i}(\mathbf r)
= \int_\Omega \omega_{\alpha i}(\mathbf r-\mathbf r')\rho_i(\mathbf r')\,d\mathbf r'.
```

For mixtures, include sums over bead/component index where appropriate:

```math
n_\alpha(\mathbf r)
= \sum_i \omega_{\alpha i} * \rho_i.
```

### Functional derivative chain rule

Derive:

```math
\delta F_{\mathrm{res}}
= \int \sum_\alpha
\frac{\partial f_{\mathrm{res}}}{\partial n_\alpha(\mathbf r)}
\delta n_\alpha(\mathbf r)\,d\mathbf r.
```

Using:

```math
\delta n_\alpha(\mathbf r)
= \sum_i \int \omega_{\alpha i}(\mathbf r-\mathbf r')
\delta\rho_i(\mathbf r')\,d\mathbf r',
```

obtain:

```math
\frac{\delta F_{\mathrm{res}}}{\delta \rho_i(\mathbf r')}
= \sum_\alpha \int
\frac{\partial f_{\mathrm{res}}}{\partial n_\alpha(\mathbf r)}
\omega_{\alpha i}(\mathbf r-\mathbf r')\,d\mathbf r.
```

Then map this to code:

```text
evaluate_field!      : ρ -> n = ω * ρ
ForwardDiff.gradient!: n[k,:,:] -> ∂f/∂n at one grid point
integrate_field!     : ∂f/∂n -> δF/δρ by convolution/integration with weights
```

### Discrete form

Include the grid-discrete version:

```math
F_{\mathrm{res}} \approx \sum_k f_{\mathrm{res}}(\mathbf n_k)\Delta V,
```

```math
n_{\alpha i,k} \approx \sum_l \omega_{\alpha i,k-l}\rho_{i,l}\Delta V,
```

```math
\frac{\partial F_{\mathrm{res}}}{\partial \rho_{i,l}}
\approx \sum_{k,\alpha}
\frac{\partial f_{\mathrm{res}}(\mathbf n_k)}{\partial n_{\alpha i,k}}
\omega_{\alpha i,k-l}\Delta V.
```

## PC-SAFT-specific content to verify and explain

When the PC-SAFT source has this decomposition:

```julia
f_res(system, model::PCSAFTModel, n) =
    f_hs(system, model, n2, n3, n4) +
    f_hc(system, model, n1, n5, n6) +
    f_disp(system, model, n7) +
    f_assoc(system, model, n2, n3, n4)
```

explain that the local residual free-energy density is decomposed as:

```math
f_{\mathrm{res}}
= f_{\mathrm{hs}} + f_{\mathrm{hc}} + f_{\mathrm{disp}} + f_{\mathrm{assoc}}.
```

For each implemented term, show both formula and code mapping. At minimum, if present, include the hard-chain and dispersion expressions.

### Hard-chain term from code

If the code computes:

```julia
ζ₃ = 1/8 * sum_i m_i ρ̄_{hc,i}
ζ₂ = 1/8 * sum_i m_i ρ̄_{hc,i}/d_i
λ_i = _λ[i] / (2*d_i)
ydd_i = 1/(1-ζ₃) + 1.5*d_i*ζ₂/(1-ζ₃)^2 + 0.5*d_i^2*ζ₂^2/(1-ζ₃)^3
f_i = -ρhc[i]*(m[i]-1)*log(ydd_i*λ_i/ρhc[i])
```

write the corresponding formula:

```math
\zeta_3 = \frac{1}{8}\sum_i m_i\bar\rho_{hc,i},
\qquad
\zeta_2 = \frac{1}{8}\sum_i \frac{m_i\bar\rho_{hc,i}}{d_i},
```

```math
\lambda_i = \frac{\Lambda_i}{2d_i},
```

```math
y_i^{dd}
= \frac{1}{1-\zeta_3}
+ \frac{3d_i\zeta_2}{2(1-\zeta_3)^2}
+ \frac{d_i^2\zeta_2^2}{2(1-\zeta_3)^3},
```

```math
f_{hc}
= -\sum_i \rho_{hc,i}(m_i-1)
\log\left(\frac{y_i^{dd}\lambda_i}{\rho_{hc,i}}\right).
```

State that this is the chain-connectivity contribution in the repository's weighted-density PC-SAFT implementation.

### Dispersion term from code

If the code computes:

```julia
ρ̄z = ρ̄ * 3 / (4*ψ^3*π*d^3)
Σρ̄ = sum(ρ̄z)
m̄ = dot(ρ̄z,m)/Σρ̄
η = π/6 * sum_i m_i ρ̄z_i d_i^3
m2ϵσ3₁, m2ϵσ3₂ = Clapeyron.m2ϵσ3(...)
C₁ = ...
I₁ = I(model,m̄,η,1)
I₂ = I(model,m̄,η,2)
f_disp = -2πΣρ̄²I₁m2ϵσ3₁ - πΣρ̄²m̄I₂m2ϵσ3₂/C₁
```

write the corresponding formula:

```math
\bar\rho^z_i
= \bar\rho_i\frac{3}{4\psi^3\pi d_i^3},
\qquad
\bar m = \frac{\sum_i \bar\rho^z_i m_i}{\sum_i \bar\rho^z_i},
```

```math
\eta = \frac{\pi}{6}\sum_i m_i\bar\rho^z_i d_i^3,
```

```math
f_{disp}
= -2\pi \bar\rho^2 I_1(\bar m,\eta)M_1
- \pi \bar\rho^2 \bar m I_2(\bar m,\eta)\frac{M_2}{C_1},
```

where `M1, M2` correspond to the two values returned by `Clapeyron.m2ϵσ3` in the code. Use the exact names from the code in the report.

## Autodiff section requirements

Find the exact line similar to:

```julia
ForwardDiff.gradient!(@view(δf[k...,:,:]), f, @view(n[k...,:,:]), cache)
```

Explain it as:

```math
\mathbf g_k
= \nabla_{\mathbf n_k} f_{\mathrm{res}}(\mathbf n_k)
```

where:

```math
g_{k,\alpha i}
= \frac{\partial f_{\mathrm{res}}(\mathbf n_k)}
{\partial n_{\alpha i,k}}.
```

Important: say this is **not** full AD of `F_res(system, ρ)` with respect to the entire density array. It is local forward-mode AD of one small function `f(n_local)` at each grid point. The full functional derivative is completed by `integrate_field!`.

Also explain CPU/GPU status:

- `Threads.@threads` means Julia CPU threads are used over grid points.
- `ForwardDiff.gradient!` is a CPU-side forward-mode AD call unless the repository has explicitly wrapped it in a GPU kernel, which this code pattern does not show.
- `synchronize(backend)` around this region suggests device work may happen before/after, but the AD loop itself is host threaded.

## GPU migration section requirements

Analyze at least these functions:

```text
F_res
δFδρ_res!
preallocate
evaluate_field!
integrate_field!
f_res(::PCSAFTModel, ...)
f_hc
f_disp
f_assoc
f_hs
```

Classify changes by file:

### Likely changes in the core model/free-energy file

This is the file containing `δFδρ_res!`. Changes likely include:

1. Remove or bypass:

   ```julia
   Threads.@threads for kk in CartesianIndices(ngrid)
       ForwardDiff.gradient!(...)
   end
   ```

2. Replace it with either:

   - a GPU kernel that computes `δf[k,:,:]` for each grid point, or
   - an analytic gradient implementation called from a GPU kernel, or
   - an Enzyme/KernelAbstractions-based AD kernel if it supports the actual code path.

3. Ensure `δf`, `n`, `fft_buf`, and caches stay on device without CPU `copyto!` round trips.

4. Rework `cache_pool`, because CPU `ForwardDiff` chunk/cache objects are not GPU-kernel compatible.

### Likely changes in the PC-SAFT model file

This is the file containing `f_res(::PCSAFTModel, ...)`. Changes likely include:

1. Make `f_res`, `f_hc`, `f_disp`, `f_hs`, and `f_assoc` GPU-callable.

2. Avoid dynamic allocations such as `similar(...)` inside functions called per grid point, unless they are stack/static arrays compatible with the GPU backend.

3. Avoid CPU-only high-level calls inside GPU kernels.

4. Check calls into Clapeyron, especially:

   ```julia
   Clapeyron.m2ϵσ3(...)
   Clapeyron.PCSAFTconsts.corr1
   Clapeyron.PCSAFTconsts.corr2
   d(model,...)
   onevec(model)
   ```

   Verify whether these are isbits/GPU-compatible and whether they allocate.

5. Consider implementing an explicit analytic gradient for PC-SAFT terms if AD on GPU is not robust.

## Minimal benchmark script requirements

The benchmark file must be saved as:

```text
benchmark/minima.jl
```

Use this structure, but replace constructor names with the actual API discovered from the repository:

```julia
#!/usr/bin/env julia

using cDFT
using Clapeyron
using Printf
using LinearAlgebra

# Adjustable grid count:
const NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

# Physical state:
const T = 300.0
const p = 1.0e5

# Example component. Replace with a component known to work in Clapeyron/PCSAFT.
components = ["methane"]
model = PCSAFT(components)

# TODO: Replace with actual structure constructor from this repository.
# Example target shape:
# structure = DFTStructure(
#     ngrid=(NGRID,),
#     bounds=[0.0 20.0],
#     conditions=(p,T),
#     ρbulk=ρbulk,
# )

# TODO: Build system using actual API.
# system = DFTSystem(model, structure; device=CPU())

# TODO: Initial density profile.
# ρ0 = fill(ρbulk[1], NGRID, 1)

# Warm-up and benchmark targets:
# F = F_res(system, ρ0)
# dF = δFδρ_res(system, ρ0)
# @printf("NGRID=%d F_res=%.8e norm(dF)=%.8e\n", NGRID, F, norm(vec(dF)))

# TODO: If minimization API exists, call it here.
# result = minimize(system, ρ0)
# @show result
```

The report must include whether the script runs successfully. If it does not run, include the exact error and the missing API/function to fix.

## Style requirements for `benchmark/current.md`

- Use Markdown headings.
- Include equations in LaTeX blocks.
- Include code snippets only where they directly clarify the code path.
- Do not write vague statements like “the code uses PC-SAFT.” Instead write “`path/to/file.jl:function_name` computes ...”.
- Distinguish confirmed source-code facts from inferred interpretation.
- If a formula is inferred from code, say “From the implementation, this corresponds to ...”.
- If the repository has tests or examples, mention how to run the benchmark and how to verify output.

## Final checklist before finishing

Before completing the task, verify that:

- [ ] `benchmark/current.md` exists.
- [ ] `benchmark/minima.jl` exists, or the report clearly explains why it could not be created.
- [ ] `current.md` cites exact file paths and function names.
- [ ] `current.md` contains the full functional derivative derivation.
- [ ] `current.md` identifies the `ForwardDiff.gradient!` call and explains CPU-side AD.
- [ ] `current.md` describes the GPU migration plan and specific files/functions to modify.
- [ ] Any benchmark code was run, or the report includes the exact reason it could not be run.