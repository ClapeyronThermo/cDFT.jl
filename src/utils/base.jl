"""
    free_energy(system::DFTSystem)

Obtain the total free energy of the system. This is done by summing the ideal and residual free energies.

The output is a scalar of units J.
"""
function free_energy(system::DFTSystem)
    return F_ideal(system)+F_res(system)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))
