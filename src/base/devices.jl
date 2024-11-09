abstract type Device end

"""
    DFTOptions(device::Device, solver::Solvers.AbstractFixPoint)

A struct which includes all the settings that need to be set for the convergence algorithms and devices used:
- `device`: Specification of either CPU (pinned or un-pinned) or GPU devices. (unpinned CPU by default)
- `solver`: Specification of the solver type and solver settings used. Must be a fixed-point method. (`AndersonFixPoint` by default)
Example usage:
```julia
julia> options = DFTOptions()

julia> using ThreadPinning

julia> options = DFTOptions(CPU(4, [0,1,12,13]))
```
"""
struct DFTOptions{D,S}
    device::D
    solver::S
end

function DFTOptions(device::Device)
    return DFTOptions(device,AndersonFixPoint())
end

function DFTOptions()
    return DFTOptions(CPU(),AndersonFixPoint())
end

"""
    CPU(ncpu::Int, pinning::Bool, device_ids::Vector{Union{Nothing,Int}})

A struct containing information regarding the CPU settings.
- `ncpu`: Number of CPU threads used
- `pinning`: Specifies whether pinning is used or not.
- `device_ids`: When CPUs are pinned, specify the ID of those threads.
"""
struct CPU <: Device 
    ncpu::Int
    pinning::Bool
    device_ids::Vector{Union{Nothing,Int}}
end

struct GPU <: Device
    ngpu::Int
    device_ids::Vector{Union{Nothing,Int}}
end

function CPU() 
    ncpu = Threads.nthreads()
    return CPU(ncpu,false,[])
end

function CPU(ncpu::Int) 
    if ncpu != Threads.nthreads()
        throw(error("Number of CPUs requested is not equal to the number of available threads."))
    end
    return CPU(ncpu,false,[])
end

function GPU(ngpu::Int,device_ids::Vector{Int}) 
    throw(error("Please load CUDA.jl to use GPU."))
end

export CPU, GPU, DFTOptions