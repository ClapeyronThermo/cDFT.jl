# PC-SAFT cDFT Technical Report

## 1. PC-SAFT cDFT theory and code path

### Functional Definition
The residual free-energy functional $F_{\mathrm{res}}$ is defined as the integral of the local residual free-energy density $f_{\mathrm{res}}$ over the domain $\Omega$:
```math
F_{\mathrm{res}}[\rho] = \int_\Omega f_{\mathrm{res}}(\mathbf n(\mathbf r))\,d\mathbf r.
```
In `src/models/models.jl:F_res`, this is implemented as:
```julia
for kk in CartesianIndices(ngrid)
    k = Tuple(kk)
    ϕ[k...] = f(@view(n[k...,:,:]))
end
return ∫(ϕ,dz)
```
where `f` is the local free-energy density function and `dz` is the grid spacing.

### Weighted Density Construction
The weighted densities $\mathbf n(\mathbf r)$ are computed by convolving the density profiles $\rho_i(\mathbf r)$ with weight functions $\omega_{\alpha i}$:
```math
n_{\alpha}(\mathbf r) = \sum_i (\omega_{\alpha i} * \rho_i)(\mathbf r) = \sum_i \int_\Omega \omega_{\alpha i}(\mathbf r-\mathbf r')\rho_i(\mathbf r')\,d\mathbf r'.
```
In the code, this is handled by `evaluate_field!` (in `src/fields/weighted_densities.jl`), which iterates over fields and components, calling `convolve!`.

For PC-SAFT, the weight functions are defined in `src/models/DFT/SAFT/PCSAFT/PCSAFT.jl:get_fields`. Seven types of weighted densities are used:
1. `n1` ($n_{\rho}$): Unweighted density $\rho_i$.
2. `n2` ($n_0^{0.5d}$): Scalar weight, radius $0.5d$.
3. `n3` ($n_3^{0.5d}$): Scalar weight, radius $0.5d$.
4. `n4` ($n_{v}^{0.5d}$): Vector weight, radius $0.5d$.
5. `n5` ($n_3^d$): Scalar weight, radius $d$.
6. `n6` ($n_0^d$): Scalar weight, radius $d$.
7. `n7` ($n_3^{\psi d}$): Scalar weight, radius $\psi d$.

### Local Residual Free-Energy Density Decomposition
The PC-SAFT local residual free-energy density $f_{\mathrm{res}}$ is decomposed into four parts:
```math
f_{\mathrm{res}} = f_{\mathrm{hs}} + f_{\mathrm{hc}} + f_{\mathrm{disp}} + f_{\mathrm{assoc}}.
```
This is implemented in `src/models/DFT/SAFT/PCSAFT/PCSAFT.jl:f_res`.

#### Hard-Sphere Term ($f_{\mathrm{hs}}$)
Implemented in `src/models/DFT/FMT.jl:f_hs`. It uses Fundamental Measure Theory (FMT) with the Yu-Wu functional.
- **Input**: `n2` ($n_0$), `n3` ($n_3$), `n4` ($n_v$).
- **Output**: Scalar free-energy density.
- **Formula**:
```math
f_{\mathrm{hs}} = -n_0 \log(1-n_3) + \frac{n_1 n_2 - \mathbf n_{v1} \cdot \mathbf n_{v2}}{1-n_3} + \dots
```

#### Hard-Chain Term ($f_{\mathrm{hc}}$)
Implemented in `src/models/DFT/SAFT/PCSAFT/PCSAFT.jl:f_hc`.
- **Input**: `ρhc` (unweighted $\rho$), `ρ̄hc` ($n_3$ type), `_λ` ($n_0$ type).
- **Formula**:
```math
\zeta_3 = \frac{1}{8}\sum_i m_i\bar\rho_{hc,i}, \quad \zeta_2 = \frac{1}{8}\sum_i \frac{m_i\bar\rho_{hc,i}}{d_i}
```
```math
y_i^{dd} = \frac{1}{1-\zeta_3} + \frac{3d_i\zeta_2}{2(1-\zeta_3)^2} + \frac{d_i^2\zeta_2^2}{2(1-\zeta_3)^3}
```
```math
f_{hc} = -\sum_i \rho_{hc,i}(m_i-1) \log\left(\frac{y_i^{dd}\lambda_i}{\rho_{hc,i}}\right)
```

#### Dispersion Term ($f_{\mathrm{disp}}$)
Implemented in `src/models/DFT/SAFT/PCSAFT/PCSAFT.jl:f_disp`.
- **Input**: `ρ̄` ($n_7$, i.e., $n_3$ type with width $\psi d$).
- **Formula**:
```math
\bar\rho^z_i = \bar\rho_i\frac{3}{4\psi^3\pi d_i^3}, \quad \eta = \frac{\pi}{6}\sum_i m_i\bar\rho^z_i d_i^3
```
```math
f_{disp} = -2\pi (\sum \bar\rho^z_i)^2 I_1 M_1 - \pi (\sum \bar\rho^z_i)^2 \bar m I_2 \frac{M_2}{C_1}
```

## 2. Minimal 1D PC-SAFT benchmark

The benchmark script is located at `benchmark/minima.jl`. It constructs a 1D PC-SAFT model for methane, initializes a uniform density profile, evaluates the free energy and its derivative, and runs the `converge!` solver.

### Execution Trace
```bash
julia --project=. benchmark/minima.jl 128
```
Output:
```text
NGRID=128 F_res=-3.26354557e+14 norm(dF)=4.16210764e-02
Final density (first grid point): 40.16461735
```

## 3. CPU autodiff step

### Identification
The automatic differentiation (AD) occurs in `src/models/models.jl:δFδρ_res!`:
```julia
Threads.@threads for kk in CartesianIndices(ngrid)
    k = Tuple(kk)
    cache = take!(cache_pool)
    ForwardDiff.gradient!(@view(δf[k...,:,:]), f, @view(n[k...,:,:]), cache)
    put!(cache_pool, cache)
end
```
- **AD Type**: Local forward-mode AD using `ForwardDiff.jl`.
- **Mathematical Derivative**: $\mathbf g_k = \nabla_{\mathbf n_k} f_{\mathrm{res}}(\mathbf n_k)$, i.e., the derivative of the local free-energy density with respect to the local weighted densities.
- **Input Array Shape**: `@view(n[k...,:,:])` which is a 2D array of shape `(NF, NB)` where `NF` is the number of fields and `NB` is the number of beads.
- **Output Array Shape**: `@view(δf[k...,:,:])` of the same shape `(NF, NB)`.
- **Execution Device**: CPU threads (`Threads.@threads`).

### Functional Derivative Chain Rule
The full functional derivative $\frac{\delta F_{\mathrm{res}}}{\delta \rho_i(\mathbf r')}$ is computed using the chain rule:
```math
\frac{\delta F_{\mathrm{res}}}{\delta \rho_i(\mathbf r')} = \sum_\alpha \int \frac{\partial f_{\mathrm{res}}}{\partial n_\alpha(\mathbf r)} \omega_{\alpha i}(\mathbf r-\mathbf r')\,d\mathbf r.
```
In the code, the integration part is performed by `integrate_field!` (in `src/fields/weighted_densities.jl`), which takes the local gradients $\partial f / \partial n_\alpha$ and convolves them with the weight functions $\omega_{\alpha i}$.

## 4. GPU migration plan

### Blockers and Required Changes
The current implementation is heavily CPU-bound due to the following factors:
1. **Host-side Loop**: `Threads.@threads` in `δFδρ_res!` must be replaced by a GPU kernel (e.g., using `CUDA.jl` or `KernelAbstractions.jl`).
2. **CPU AD**: `ForwardDiff.gradient!` with `GradientConfig` caches cannot run inside a standard GPU kernel.
3. **FFT Backend**: `FFTW` is used for convolutions in `evaluate_field!` and `integrate_field!`. This must be switched to `CUFFT`.
4. **Memory Transfers**: While `Adapt.adapt` is used, the `ForwardDiff` loop involves views and potential host-device synchronizations that would kill performance.

### Implementation Plan
1. **Core Model File (`src/models/models.jl`)**:
   - Replace the `Threads.@threads` loop in `δFδρ_res!` with a custom GPU kernel that computes the gradient at each grid point.
   - Replace `cache_pool` with device-compatible memory buffers.
2. **PC-SAFT Model File (`src/models/DFT/SAFT/PCSAFT/PCSAFT.jl`)**:
   - Ensure all functions called by `f_res` (`f_hs`, `f_hc`, `f_disp`, etc.) are `@inline` and contain only GPU-compatible operations (no allocations, no CPU-only Clapeyron calls).
   - Verify `Clapeyron.m2ϵσ3` and `I` function compatibility.
3. **Differentiation Strategy**:
   - **Option A (Analytic)**: Hand-code the analytic derivatives $\partial f / \partial n_\alpha$. This is the most performant but hardest to maintain.
   - **Option B (Enzyme.jl)**: Use `Enzyme.jl` for high-performance AD that can generate GPU kernels from Julia code.
   - **Option C (Dual Numbers)**: Re-implement a lightweight version of forward-mode AD using `ForwardDiff.Dual` directly inside a GPU kernel without the heavy `GradientConfig`.

### Practical Steps
- Audit `f_res` for any non-`isbits` types or dynamic allocations.
- Replace `plan_fft!` in `weighted_densities.jl` with a backend-agnostic version that uses `CUDA.plan_fft!` when on GPU.
- Implement a single kernel that wraps the `f_res` call and its differentiation to minimize kernel launch overhead and memory traffic.
