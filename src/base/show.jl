import Clapeyron: show_pairs

function Base.show(io::IO,::MIME"text/plain",system::DFTSystem)
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

function Base.show(io::IO,::MIME"text/plain", system::DGTSystem)
    n = length(system.model)
    print(io,"DGTSystem with ",n," component")
    n > 1 && print(io,"s")
    println(io,":")
    
    #if hasfield(typeof(system.model),:components)
    #    show_pairs(io,system.model.components, prekey=" ", pair_separator=", ")
    #end
    print(io," model: ")
    show(io,system.model)
    println(io)

    print(io," gradient: ")
    show(io,system.gradient)
    println(io)
    println(io," structure: "*string(typeof(system.structure)))
    print(io," device: "*string(typeof(system.options.device)))
end

function Base.show(io::IO,mime::MIME"text/plain", system::GradientModel)
    return Clapeyron.eosshow(io,mime,system)
end

function Base.show(io::IO, system::GradientModel)
    return Clapeyron.eosshow(io,system)
end