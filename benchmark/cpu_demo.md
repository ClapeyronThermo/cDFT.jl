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

The efficiency of cDFT.jl on GPUs comes from its specialized handling of the Helmholtz free energy functional derivatives.

### Grid-wise Autodiff with Enzyme.jl
Instead of computing the full functional derivative through global automatic differentiation or manual analytical derivation (which is complex for models like PC-SAFT), we use:
- **Enzyme.jl**: A high-performance compiler plugin that performs reverse-mode AD directly on the GPU kernels.
- **In-place Gradients**: The gradient of the local free energy density with respect to weighted densities is computed at each grid point in parallel.
- **Zero Host-Device Transfer**: All computations, including the AD and field integrations (using FFTs via CUDA.jl), happen entirely on the GPU.
- **Kernel-level Optimization**: By passing system parameters as Const to the Enzyme kernel, we minimize overhead and maximize throughput.

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
