module PlotscDFTExt

using cDFT
using Plots

function Plots.plot(system::cDFT.DFTSystem)
    profiles = system.profiles
    structure = system.structure
    model = system.model
    nb = length(profiles)

    bounds = structure.bounds

    z = LinRange(bounds[1],bounds[2],structure.ngrid*10)
    L = cDFT.length_scale(model)

    plt = Plots.plot(grid=:off,
                    framestyle=:box,
                    foreground_color_legend = nothing,
                    xtickfontsize=12,
                    ytickfontsize=12,
                    xlabelfontsize=14,
                    ylabelfontsize=14,
                    legend_font=font(12))

    ymax = 0.
    for i in 1:nb
        species_id = system.species.species_id[i]
        group_id = system.species.group_id[i]
        species = model.groups.flattenedgroups[group_id]
        Mw = model.params.Mw[group_id]
        Plots.plot!(plt,z./L,profiles[i].(z)*Mw/1e3,label="$species",linewidth=3)
        ymax = max(ymax,maximum(profiles[i].density)*Mw/1e3)
    end
    Plots.xlims!(plt,(bounds[1],bounds[2])./L)
    Plots.ylims!(plt,(0,1.1*ymax))
    if typeof(system.structure) <: cDFT.DFTStructure1DSphr 
        Plots.xlabel!(plt,"r / σ")
    elseif typeof(system.structure) <: cDFT.DFTStructure1DCart
        Plots.xlabel!(plt,"z / σ")
    end

    Plots.ylabel!(plt,"ρ / (kg/m³)")
    return plt
end

end