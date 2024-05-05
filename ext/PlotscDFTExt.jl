module PlotscDFTExt

using cDFT
using Plots

function Plots.plot(system::cDFT.DFTSystem; x_units=:normalized, y_units=:mass)
    profiles = system.profiles
    structure = system.structure
    model = system.model
    species = system.species
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
    species_id = 1
    bead_id = 1
    for i in 1:nb
        if species[species_id].nbeads > 1
            species_name = model.components[species_id]
            group_id = species[species_id].bead_id[bead_id]
            group_name = model.groups.flattenedgroups[group_id]
            name = "$species_name $group_name $bead_id"
            Mw = model.params.Mw[group_id]
            norm_const = model.params.segment[group_id]*species[species_id].size[bead_id]^3*cDFT.N_A
        else
            species_name = model.components[species_id]
            name = "$species_name"
            Mw = model.params.Mw[species_id]
            norm_const = model.params.segment[species_id]*species[species_id].size[species_id]^3*cDFT.N_A
        end

        if x_units == :normalized
            X = z./L
        elseif x_units == :angstrom
            X = z.*1e10
        elseif x_units == :nanometer
            X = z.*1e9
        else
            X = z
        end

        if y_units == :normalized
            Y = profiles[i].(z).*norm_const
            y_norm = "σ³"
        elseif y_units == :mass
            Y = profiles[i].(z).*Mw/1e3
            y_norm = " / (kg/m³)"
        else
            Y = profiles[i].(z)
            y_norm = " / (mol/m³)"
        end
        
        Plots.plot!(plt,X,Y,label="$name",linewidth=3)
        ymax = max(ymax,maximum(Y))

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    

    if x_units == :normalized
        Plots.xlims!(plt,(bounds[1],bounds[2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Plots.xlims!(plt,(bounds[1],bounds[2]).*1e9)
        x_norm = "nm"
    else
        Plots.xlims!(plt,(bounds[1],bounds[2]))
        x_norm = "m"
    end

    Plots.ylims!(plt,(0,1.1*ymax))
    if typeof(system.structure) <: cDFT.DFTStructure1DSphr 
        Plots.xlabel!(plt,"r / "*x_norm)
    elseif typeof(system.structure) <: cDFT.DFTStructure1DCart
        Plots.xlabel!(plt,"z / "*x_norm)
    end

    if y_units == :normalized
        Plots.ylabel!(plt,"ρσ³")
    elseif y_units == :mass
        Plots.ylabel!(plt,"ρ / (kg/m³)")
    else
        Plots.ylabel!(plt,"ρ / (mol/m³)")
    end

    return plt
end

end