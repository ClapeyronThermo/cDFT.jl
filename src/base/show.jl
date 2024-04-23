import Clapeyron: show_pairs

function Base.show(io::IO,system::DFTSystem)
    println(io,"DFTSystem")
    println(io,"  model: "*string(typeof(system.model)))
    if hasfield(typeof(system.model),:components)
        length(system.model) == 1 && print(io, "\t with 1 component:")
        length(system.model) > 1 && print(io, "\t with ", length(system.model), " components:")
        show_pairs(io,system.model.components, prekey=" ", pair_separator=", ")
    end

    println(io,"\n  structure: "*string(typeof(system.structure)))
    println(io,"  device: "*string(typeof(system.options.device)))
end