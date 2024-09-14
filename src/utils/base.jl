"""
    free_energy(system::DFTSystem)

Obtain the total free energy of the system. This is done by summing the ideal and residual free energies.

The output is a scalar of units J.
"""
function free_energy(system::DFTSystem)
    return F_ideal(system)+F_res(system)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))

macro chain(component, args...)
    quote
        if hasfield(typeof(system.model), :groups)
            system.model.groups.i_groups[$(component)]
        else
            $(component)
        end
    end |> esc
end