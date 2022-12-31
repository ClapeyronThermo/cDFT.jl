module ClapeyronDFT

using ForwardDiff
using Clapeyron
using Clapeyron: d, N_A, k_B
using Clapeyron: Solvers, log, sqrt, log1p
using Clapeyron: PCSAFTModel


include("utils/types.jl")
include("utils/base.jl")
include("utils/profiles.jl")
include("utils/integrals.jl")
include("utils/initial.jl")

include("models/weights.jl")
include("models/FMT.jl")
include("models/HC.jl")
include("models/PCSAFT.jl")


end