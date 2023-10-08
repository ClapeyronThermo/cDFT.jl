module cDFT

using LinearAlgebra
using ForwardDiff, Optim, FixedPointAcceleration, NLSolvers
using Clapeyron
using Clapeyron: d, N_A, k_B, R̄
using Clapeyron: @comps
using Clapeyron: Solvers, log, sqrt, log1p, PackedVofV, sparse, SparseMatrixCSC
using Clapeyron: PCSAFTModel, PPCSAFTModel, BasicIdealModel
using Clapeyron: assoc_similar, assoc_matrix_solve, assoc_options, assoc_pair_length
using Clapeyron: issite, compute_index, complement_index, indices
import Clapeyron.Solvers
using Clapeyron: SingleComp
using StaticArrays

include("utils/types.jl")
include("utils/base.jl")
include("utils/profiles.jl")
include("utils/initial.jl")
include("utils/integrals.jl")
include("utils/anderson.jl") #TODO, port this to Clapeyron

include("models/BasicIdeal.jl")

include("models/weights.jl")
include("models/FMT.jl")
include("models/PCSAFT.jl")
include("models/PPCSAFT.jl")

include("methods/profiles.jl")
include("methods/inhomogeneous_free_energy.jl")
include("methods/surface_tension.jl")
include("methods/interfacial_tension.jl")

end