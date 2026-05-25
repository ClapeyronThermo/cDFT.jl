# Project Proposal: GPU Acceleration of PC-SAFT Density Functional Theory

Zhuolin He, CCE, Caltech

zhe3@caltech.edu

## Summary
This project aims to accelerate the core residual free-energy functional derivative (`δFδρ_res`) of the `cDFT.jl` package using GPUs. By offloading the grid-wise Automatic Differentiation (AD), we expect to achieve significant speedups for large 1D systems, and can potentially extend to 2D/3D problems.

## Background Information
Classical Density Functional Theory (cDFT) is a powerful tool for predicting the equilibrium structure and thermodynamic properties of inhomogeneous fluids. The `cDFT.jl` package implements several advanced functionals, including the Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT). 

The current implementation relies on `ForwardDiff.jl` to compute local derivatives of the free-energy density at each grid point. While efficient on CPUs via multi-threading, these operations are highly parallelizable and well-suited for GPU architectures, which could reduce computation times from minutes to seconds for complex 3D interfacial systems.

## Computation details
The residual Helmholtz free energy is defined as $F_{\text{res}}[\rho] = \int f_{\text{res}}(\mathbf{n}(\mathbf{r})) d\mathbf{r}$, where $\mathbf{n}(\mathbf{r})$ are weighted densities. The functional derivative involves:
1. **Weighted Density Evaluation**: $\mathbf{n} = \omega * \rho$ (Convolution via FFT).
2. **Local Derivative Calculation**: $\frac{\partial f_{\text{res}}}{\partial n_\alpha}$ at every grid point using forward-mode AD (`ForwardDiff`).
3. **Back-projection/Integration**: $\frac{\delta F_{\text{res}}}{\delta \rho} = \sum_\alpha \omega * \frac{\partial f_{\text{res}}}{\partial n_\alpha}$.

<!-- Here, we noticed that the function $f_{\text{res}}$ is not analytical when there are multiple associating species (for example, water and methanol mixture), so we only consider the systems that have up to 1 associating species. -->

## Project explanation
This project will migrate the bottleneck, which is the grid-wise local derivative calculation, to GPU. We will leverage `CUDA.jl` for memory management and `Enzyme.jl` (and/or other tools) for high-performance AD directly on GPU kernels. This shift is critical for enabling large-scale molecular simulations that are currently computationally prohibitive.

## Questions to address

### Previous GPU implementations?
Currently, the `cDFT.jl` package successfully utilizes GPU acceleration for convolutions, but the AD component remains CPU-bound though using multi-threading. There are no existing GPU implementations of the functional derivative in `cDFT.jl`, making this a novel contribution. In the Julia ecosystem, `Enzyme.jl` has shown promise for differentiating physical kernels, but it is still not implemented in the code.

### Technical challenges and Problem to solve
- **Compatibility**: Ensuring that all components (especially the caching mechanisms of `ForwardDiff.jl` or `Enzyme.jl`) of the functional derivative (especially the AD logic) are compatible with GPU execution, which may require significant refactoring.
- **Memory Management**: Efficiently managing GPU memory, especially for large grid sizes, and minimizing data transfer between host and device.
- **Standardization**: Modifying the structure to support both CPU(can be multi-threaded) and GPU backends without code duplication.

## Deliverables and goals
1. **GPU-Enabled Functional Derivative**: A working implementation of `δFδρ_res` that runs on CPUs and GPUs, specified by `system.options.device`.
2. **Accuracy Verification**: The GPU implementation must match the CPU baseline (if the system is homogeneous, it must match the CPU   Clapeyron bulk $\mu$) within a tolerance.
3. **Performance Benchmarks**: Demonstrated speedup over the CPU baseline for $N > 10^5$, and find out the bottleneck for the GPU implementation.

## Week-by-week timeline
- **Week 1: CUDA in Julia**: Familiarize with `CUDA.jl` and set up a simple GPU kernel for testing. Testing rosenbrock function with `Enzyme.jl` and `ForwardDiff.jl` to understand the AD performance on CPU and GPU.
- **Week 2: GPU AD Implementation**: Investigate the structure of `cDFT.jl` and isolate the AD component. Begin implementing the GPU version of the local derivative calculation using `Enzyme.jl` or `ForwardDiff.jl`. Write unit tests to verify correctness against the CPU implementation.
- **Week 3 and 4: Integration and Optimization**: Integrate the GPU AD into the full functional derivative calculation. Optimize memory usage and kernel performance. Benchmark against the CPU version for various grid sizes and analyze results.
