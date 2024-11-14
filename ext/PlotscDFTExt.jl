module PlotscDFTExt

using cDFT
using Plots
import Plots: Colors


function Plots.plot(system::cDFT.DFTSystem, profiles; x_units=:normalized, y_units=:normalized)
    return Plots.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units)
end

function Plots.plot(system::cDFT.DFTSystem, structure::cDFT.DFTStructure1DCart, profiles; x_units=:normalized, y_units=:mass)
    structure = system.structure
    model = system.model
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    z = uniform_range(structure)
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
    for i in cDFT.@comps
        for k in cDFT.@chain(i)
            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*cDFT.N_A
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*cDFT.N_A
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
                Y = profiles[:,k].*norm_const
                y_norm = "σ³"
            elseif y_units == :mass
                Mw = model.params.Mw[k]
                Y = profiles[:,k].*Mw/1e3
                y_norm = " / (kg/m³)"
            elseif y_units == :angstrom
                Y = profiles[:,k].*cDFT.N_A/1e30
                y_norm = " / (kg/m³)"
            else
                Y = profiles[:,k]
                y_norm = " / (mol/m³)"
            end
        
            Plots.plot!(plt,X,Y,label="$name",linewidth=3)
            ymax = max(ymax,maximum(Y))
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

    Plots.plot!(plt,legend=:topleft)

    return plt
end

function Plots.plot(system::cDFT.DFTSystem, structure::cDFT.DFTStructure2DCart, profiles; x_units=:normalized, y_units=:normalized)
    colors = palette(:tab10)
    structure = system.structure
    model = system.model
    species = system.species
    nb = length(profiles)

    bounds = structure.bounds

    x = uniform_range(structure,1)
    y = uniform_range(structure,2)
    X = zeros(length(x),length(y))
    Y = zeros(length(x),length(y))

    for i in 1:length(x)
        X[i,:] .= x[i]
    end

    for i in 1:length(y)
        Y[:,i] .= y[i]
    end

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
    for i in cDFT.@comps
        for k in cDFT.@chain(i)
            if species.nbeads[i] > 1
                species_name = model.components[i]
                group_name = model.groups.flattenedgroups[k]
                name = "$species_name $group_name"
                norm_const = model.params.segment[k]*species.size[k]^3*cDFT.N_A
            else
                species_name = model.components[i]
                name = "$species_name"
                norm_const = model.params.segment[i]*species.size[i]^3*cDFT.N_A
            end

            if x_units == :normalized
                X = x./L
            elseif x_units == :angstrom
                X = x.*1e10
            elseif x_units == :nanometer
                X = x.*1e9
            else
                X = x
            end

            if y_units == :normalized
                Y = y./L
            elseif x_units == :angstrom
                Y = y.*1e10
            elseif x_units == :nanometer
                Y = y.*1e9
            else
                Y = y
            end

            Z = profiles[:,:,k].*norm_const
            z_norm = "σ³"

            # if y_units == :normalized
            #     Z = profiles[:,k].*norm_const
            #     y_norm = "σ³"
            # elseif y_units == :mass
            #     Mw = model.params.Mw[k]
            #     Y = profiles[:,k].*Mw/1e3
            #     y_norm = " / (kg/m³)"
            # elseif y_units == :angstrom
            #     Y = profiles[:,k].*cDFT.N_A/1e30
            #     y_norm = " / (kg/m³)"
            # else
            #     Y = profiles[:,k]
            #     y_norm = " / (mol/m³)"
            # end
        
            Plots.heatmap!(plt,X,Y,Z,label="$name", c=cgrad([:white, colors[k]]))
        end
    end
    

    if x_units == :normalized
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]).*1e9)
        x_norm = "nm"
    else
        Plots.xlims!(plt,(bounds[1,1],bounds[1,2]))
        x_norm = "m"
    end

    if y_units == :normalized
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2])./L)
        y_norm = "σ"
    elseif y_units == :angstrom
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]).*1e10)
        y_norm = "Å"
    elseif y_units == :nanometer
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]).*1e9)
        y_norm = "nm"
    else
        Plots.ylims!(plt,(bounds[2,1],bounds[2,2]))
        y_norm = "m"
    end

    # if typeof(system.structure) <: cDFT.DFTStructure1DSphr 
    #     Plots.xlabel!(plt,"r / "*x_norm)
    # elseif typeof(system.structure) <: cDFT.DFTStructure1DCart
    Plots.xlabel!(plt,"x / "*x_norm)
    Plots.ylabel!(plt,"y / "*y_norm)

    # if y_units == :normalized
    #     Plots.ylabel!(plt,"ρσ³")
    # elseif y_units == :mass
    #     Plots.ylabel!(plt,"ρ / (kg/m³)")
    # else
    #     Plots.ylabel!(plt,"ρ / (mol/m³)")
    # end

    # Plots.plot!(plt,legend=:topleft)

    return plt
end

end