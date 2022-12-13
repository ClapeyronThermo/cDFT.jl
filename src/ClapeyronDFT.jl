module ClapeyronDFT

using Clapeyron
using Clapeyron: d, N_A, k_B

include("utils/types.jl")
include("utils/integrals.jl")

include("models/FMT.jl")
include("models/HC.jl")
include("models/PCSAFT.jl")

include("methods/initial.jl")

end