import Clapeyron: show_pairs

function Base.show(io::IO,system::DFTSystem)
    n = length(system.model)
    print(io,"DFTSystem with ",n," component")
    n > 1 && print(io,"s")
    println(io,":")
    
    #if hasfield(typeof(system.model),:components)
    #    show_pairs(io,system.model.components, prekey=" ", pair_separator=", ")
    #end
    print(io," model: ")
    show(io,system.model)
    println(io)
    println(io," structure: "*string(typeof(system.structure)))
    print(io," device: "*string(typeof(system.options.device)))
end