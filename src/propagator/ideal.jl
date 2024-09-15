function propagate(system::DFTSystem, propagate::IdealPropagator, δf_res)
    structure = system.structure
    ngrid = structure.ngrid
    nbeads = sum(system.species.nbeads)

    I1 = ones(Float64, ngrid, nbeads, nbeads)
    I2 = ones(Float64, ngrid, nbeads)
    return I1, I2
end