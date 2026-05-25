1. **Functional Derivative Validation in `benchmark/minima.jl`**:
   - **Implement Accuracy Check**: In `benchmark/minima.jl`, calculate the bulk residual chemical potential using Clapeyron's built-in function: `μ_bulk = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T`.
   - **Compare with dF/dρ**: Compare `μ_bulk` against the values in `dF = cDFT.δFδρ_res(system, ρ0)`. For a uniform density profile, these should match at every grid point. Print the maximum absolute error between them to verify the functional derivative implementation.

2. **Refactor Benchmark Script (`benchmark/minima.jl`)**:
   - **Update Component**: Change `components = ["methane"]` to `components = ["hexane"]`. Hexane at $300\,\text{K}, 1\,\text{atm}$ is in a liquid state, providing a more robust test case for the functional.
   - **Remove Convergence Step**: Delete the call to `cDFT.converge!(system, ρ0)`. The benchmark should focus on the cost and accuracy of a single derivative evaluation, not the solver's convergence.

3. **Performance Profiling**:
   - Add `using BenchmarkTools` to `benchmark/minima.jl`.
   - Use `@btime` to measure the execution time and memory allocations of `cDFT.δFδρ_res(system, ρ0)`. This will serve as the CPU baseline for the GPU migration.

4. **Project Proposal (`benchmark/proposal.md`)**:
   - Draft a comprehensive proposal in `./benchmark/proposal.md` following these exact requirements:
     - **Summary** (1-3 sentences)
     - **Background Information** (1-3 paragraphs)
     - **Computation details**
     - **Project explanation**
     - **Questions to address**
     - **Previous GPU implementations?**
     - **Technical challenges**
     - **Problems to solve**
     - **Deliverables and goals**
     - **Week-by-week timeline** (Duration: 4 weeks)
