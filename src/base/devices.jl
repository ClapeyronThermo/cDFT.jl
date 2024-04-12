abstract type Device end

struct DFTOptions
    device::Device
    solver::Solvers.AbstractFixPoint
end

struct CPU <: Device 
    ncpu::Int
    pinning::Bool
    device_ids::Vector{Union{Nothing,Int}}
end

struct GPU <: Device
    ngpu::Int
    device_ids::Vector{Int}
end

function CPU() 
    ncpu = Threads.nthreads()
    return CPU(ncpu,false,[])
end

function CPU(ncpu::Int) 
    return CPU(ncpu,false,[])
end

function CPU(ncpu::Int,pinning::Bool,device_ids::Vector{Int}) 
    throw(error("Please load ThreadPinning.jl to use pinning."))
end

function GPU(ngpu::Int,device_ids::Vector{Int}) 
    throw(error("Please load CUDA.jl to use GPU."))
end