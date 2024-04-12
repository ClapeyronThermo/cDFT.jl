module ThreadPinningcDFTExt

using cDFT
using ThreadPinning

function cDFT.CPU(ncpu::Int,device_ids::Vector{Int}) 
    ThreadPinning.pinthreads(device_ids)
    return CPU(ncpu,true,device_ids)
end

end