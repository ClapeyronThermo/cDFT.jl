function ∫ρdz(structure::DFTStructure1DCart,ρ::DensityProfile,span::Float64)
    span *= 2π
    ft = [ρ.density;reverse(ρ.density)]
    f = 1/ρ.mesh_size
    ω = fftfreq(structure.ngrid*2, f)
    ω1(ω) = 2*span * (ω == 0.0) + 2*sin(ω*span)/ω *(ω != 0.0)
    Ω1 = ω1.(ω)
    return real.(ifft(fft(ft).*Ω1))[1:structure.ngrid]/(2π)
end

function ∫ρz²dz(structure::DFTStructure1DCart,ρ::DensityProfile,span::Float64)
    span *= 2π
    ft = [ρ.density;reverse(ρ.density)]
    f = 1/ρ.mesh_size
    ω = fftfreq(structure.ngrid*2, f)
    ω3(ω) = 4π/ω^3*(sin(ω*span)-span*ω*cos(ω*span)) .*(ω != 0.0) + span^3/3*4π*(ω == 0.0)
    Ω3 = ω3.(ω)
    return real.(ifft(fft(ft).*Ω3))[1:structure.ngrid]/(2π)^3
end

function ∫ρzdz(structure::DFTStructure1DCart,ρ::DensityProfile,span::Float64)
    span *= 2π
    ft = [ρ.density;reverse(ρ.density)]
    f = 1/ρ.mesh_size
    ω = fftfreq(structure.ngrid*2, f)
    ω2(ω) = 4π*im/ω^2*(sin(ω*span)-span*ω*cos(ω*span)) .*(ω != 0.0) + 0.0
    Ω2 = ω2.(ω)
    return real.(ifft(fft(ft).*Ω2))[1:structure.ngrid]/(2π)^3
end