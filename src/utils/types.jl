abstract type FunctionalModel end
abstract type DFTProfile end
abstract type DensityProfile{ℂ,ρ} <: DFTProfile end
abstract type SAFTFunctionalModel <: FunctionalModel end