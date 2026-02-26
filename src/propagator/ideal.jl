function propagate(system::DFTSystem, propagator::IdealPropagator, δf_res, ρ)
    structure = system.structure
    ngrid = structure.ngrid
    nbeads = sum(system.species.nbeads)

    I1 = ones(Float64, ngrid..., nbeads, nbeads)
    I2 = ones(Float64, ngrid..., nbeads)
    return I1, I2
end