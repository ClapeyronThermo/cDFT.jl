abstract type Device end

struct DFTOptions
    device::Device
    solver::Solvers.AbstractFixPoint
end

function DFTOptions(device::Device)
    return DFTOptions(device,AndersonFixPoint())
end

function DFTOptions()
    return DFTOptions(CPU(),AndersonFixPoint())
end

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