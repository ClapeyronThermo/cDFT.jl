# CPU Demo: GPU Acceleration of PC-SAFT Density Functional Theory

Zhuolin He, CCE, Caltech

zhe3@caltech.edu

## Simulation Overview

The demo performs a 2D phase separation simulation (Vapor-Liquid Equilibrium).

- **Simulation Script**: dynamic_dft_2d_vle_hetero.jl
- **Visualization Script**: dynamic_dft_2d_viz.jl
- **Output**: trajectory_2d_vle_hetero.gif

The simulation utilizes a Uniform2DCart structure and a high-order ODE solver (ROCK2) to evolve density profiles. It demonstrates the transition from a nearly homogeneous bulk density to a phase-separated state.

![Phase Separation Simulation](trajectory_2d_vle_hetero.gif)

## Parallelization Strategy

The efficiency of cDFT.jl on GPUs is achieved by moving from a host-dependent fallback to a fully device-native implementation of the free energy functional derivatives.

### Evolution from GPU-CPU-GPU to Native GPU
Original versions of the residual Helmholtz free energy functional (`f_res`) were not compatible with GPU-based automatic differentiation (AD) or Enzyme.jl due to their reliance on complex, non-bitstype thermodynamic structures. This necessitated a **GPU-CPU-GPU** strategy:
1.  Compute weighted densities on the GPU.
2.  Transfer data back to the **CPU** to evaluate functional derivatives using standard AD.
3.  Transfer results back to the **GPU** for the next step.

### New "Lite" GPU Kernel
To eliminate this bottleneck, \`cDFT.jl\` implements a specialized **GPU-native version**:
- **f_res_lite_void_gpu**: A streamlined, bitstype-safe version of the PC-SAFT free energy density designed specifically for GPU kernels. It unrolls complex loops and uses raw arrays for maximum compatibility with Enzyme.jl.
- **Grid-wise Reverse AD**: Instead of global AD, we perform reverse-mode autodiff at each grid point using `Enzyme.autodiff_deferred`. This computes the local gradient of the free energy density with respect to all weighted densities in a single, parallel pass.
- **Global Void Pattern**: Uses specialized buffers (`f_val`, `δf_val`) to handle local function evaluations during the AD process without creating temporary allocations inside the kernel.
- **Zero-Copy Execution**: By using `preallocate_newautodiff`, all necessary buffers (weighted densities `n`, gradients `δf`, and FFT buffers) are kept in GPU memory, ensuring the entire simulation loop stays on the device.

## Test Cases and Verification

Validation is performed in ```scan.jl```, which compares the performance and accuracy of different implementations:
- **Accuracy**: The results from the new Enzyme-based autodiff kernel are verified against the original AD implementation on both CPU and GPU.
- **Performance**: Benchmarks are conducted across a range of grid sizes (up to 2^20 points) to measure the speedup of the GPU implementation over the CPU version.
- **Consistency**: The implementation ensures that new_gpu results match old_cpu results within numerical precision (< 1e-12).

## Running Instructions

To execute the demo:

1.  **Run the Example Simulation**:
    ```bash
    julia --project=.. dynamic_dft_2d_vle_hetero.jl
    ```
    *Note: Ensure you have a CUDA-compatible GPU and the benchmark/myenv environment is instantiated.*

2.  **Generate the GIF**:
    ```bash
    julia --project=.. dynamic_dft_2d_viz.jl
    ```

3.  **Perform Benchmarking GPU vs CPU**:
    ```bash
    julia --project=.. scan.jl
    ```

### Dependencies

- Please refer to the `Project.toml` in the main directory for the required Julia packages, including:
  - CUDA.jl
  - Enzyme.jl
  - HDF5.jl
  - Plots.jl
  - ...
