module MakiecDFTExt

using cDFT
using Makie

_maybe_texlabel(s, latex::Bool) = latex ? cDFT.texlabel(s) : s

function Makie.plot(system::cDFT.AbstractcDFTSystem, profiles; x_units=:normalized, y_units=:normalized, latex=false)
    return Makie.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units, latex=latex)
end

function Makie.plot(system::cDFT.AbstractcDFTSystem, structure::cDFT.DFTStructure1DCart, profiles; x_units=:normalized, y_units=:mass, latex=false)
    structure = system.structure
    model = system.model
    if model isa cDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species

    bounds = structure.bounds
    z = cDFT.uniform_range(structure)
    L = cDFT.length_scale(model)

    fig = Figure()
    ax = Axis(fig[1, 1];
        xgridvisible=false, ygridvisible=false,
        xticklabelsize=12, yticklabelsize=12,
        xlabelsize=14, ylabelsize=14)

    ymax = 0.
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

            Makie.lines!(ax, X, Y; label=_maybe_texlabel(name,latex), linewidth=3)
            ymax = max(ymax,maximum(Y))
        end
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1],bounds[2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1],bounds[2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1],bounds[2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1],bounds[2]))
        x_norm = "m"
    end

    Makie.ylims!(ax,(0,1.1*ymax))
    ax.xlabel = _maybe_texlabel("z / "*x_norm,latex)

    if y_units == :normalized
        ax.ylabel = _maybe_texlabel("ρσ³",latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt)

    return fig
end

function Makie.plot(system::cDFT.AbstractcDFTSystem, structure::Union{cDFT.DFTStructure1DSphr,cDFT.DFTStructure1DCyl}, profiles; x_units=:normalized, y_units=:mass, latex=false)
    structure = system.structure
    model = system.model
    if model isa cDFT.ElectrolyteModel
        model = model.neutralmodel
    end
    species = system.species

    bounds = structure.bounds
    z = cDFT.structure_r(structure)
    L = cDFT.length_scale(model)

    fig = Figure()
    ax = Axis(fig[1, 1];
        xgridvisible=false, ygridvisible=false,
        xticklabelsize=12, yticklabelsize=12,
        xlabelsize=14, ylabelsize=14)

    ymax = 0.
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

            Makie.lines!(ax, X, Y; label=_maybe_texlabel(name,latex), linewidth=3)
            ymax = max(ymax,maximum(Y))
        end
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1],bounds[2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1],bounds[2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1],bounds[2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1],bounds[2]))
        x_norm = "m"
    end

    Makie.ylims!(ax,(0,1.1*ymax))
    ax.xlabel = _maybe_texlabel("r / "*x_norm,latex)

    if y_units == :normalized
        ax.ylabel = _maybe_texlabel("ρσ³",latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt)

    return fig
end

function Makie.plot(system::Union{cDFT.DFTSystem,cDFT.DGTSystem}, structure::cDFT.DFTStructure2DCart, profiles; x_units=:normalized, y_units=:normalized, latex=false)
    structure = system.structure
    model = system.model
    species = system.species

    bounds = structure.bounds

    x = cDFT.uniform_range(structure,1)
    y = cDFT.uniform_range(structure,2)
    L = cDFT.length_scale(model)

    fig = Figure()
    ax = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false, aspect=Makie.DataAspect())

    colors = Makie.wong_colors()

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

            # density is always shown normalized here (σ³ units), matching the existing
            # Plots 2D method's behavior — y_units is accepted but unused for density.
            Z = profiles[:,:,k].*norm_const

            c = colors[mod1(k, length(colors))]
            csalpha = [Makie.RGBAf(c.r, c.g, c.b, 0.0), Makie.RGBAf(c.r, c.g, c.b, 1.0)]
            Makie.heatmap!(ax, X, Y, Z; colormap=csalpha, label=_maybe_texlabel(name,latex))
        end
    end

    if x_units == :normalized
        Makie.xlims!(ax,(bounds[1,1],bounds[1,2])./L)
        x_norm = "σ"
    elseif x_units == :angstrom
        Makie.xlims!(ax,(bounds[1,1],bounds[1,2]).*1e10)
        x_norm = "Å"
    elseif x_units == :nanometer
        Makie.xlims!(ax,(bounds[1,1],bounds[1,2]).*1e9)
        x_norm = "nm"
    else
        Makie.xlims!(ax,(bounds[1,1],bounds[1,2]))
        x_norm = "m"
    end

    if y_units == :normalized
        Makie.ylims!(ax,(bounds[2,1],bounds[2,2])./L)
        y_norm = "σ"
    elseif y_units == :angstrom
        Makie.ylims!(ax,(bounds[2,1],bounds[2,2]).*1e10)
        y_norm = "Å"
    elseif y_units == :nanometer
        Makie.ylims!(ax,(bounds[2,1],bounds[2,2]).*1e9)
        y_norm = "nm"
    else
        Makie.ylims!(ax,(bounds[2,1],bounds[2,2]))
        y_norm = "m"
    end

    ax.xlabel = _maybe_texlabel("x / "*x_norm,latex)
    ax.ylabel = _maybe_texlabel("y / "*y_norm,latex)

    return fig
end

function Makie.plot(system::Union{cDFT.DFTSystem,cDFT.DGTSystem}, structure::cDFT.DFTStructure3DCart, profiles; x_units=:normalized, y_units=:normalized, latex=false)
    structure = system.structure
    model = system.model
    species = system.species

    x = cDFT.uniform_range(structure,1)
    y = cDFT.uniform_range(structure,2)
    z = cDFT.uniform_range(structure,3)
    L = cDFT.length_scale(model)

    if x_units == :normalized
        X, Y, Z = x./L, y./L, z./L
        x_norm = "σ"
    elseif x_units == :angstrom
        X, Y, Z = x.*1e10, y.*1e10, z.*1e10
        x_norm = "Å"
    elseif x_units == :nanometer
        X, Y, Z = x.*1e9, y.*1e9, z.*1e9
        x_norm = "nm"
    else
        X, Y, Z = x, y, z
        x_norm = "m"
    end

    fig = Figure()
    ax = Axis3(fig[1, 1];
        aspect=:data,
        xgridvisible=false, ygridvisible=false, zgridvisible=false,
        xspinesvisible=false, yspinesvisible=false, zspinesvisible=false,
        xlabel=_maybe_texlabel("x / "*x_norm,latex),
        ylabel=_maybe_texlabel("y / "*x_norm,latex),
        zlabel=_maybe_texlabel("z / "*x_norm,latex))

    colors = Makie.wong_colors()

    for i in cDFT.@comps
        for k in cDFT.@chain(i)
            if species.nbeads[i] > 1
                norm_const = model.params.segment[k]*species.size[k]^3*cDFT.N_A
            else
                norm_const = model.params.segment[i]*species.size[i]^3*cDFT.N_A
            end

            ρk = profiles[:,:,:,k].*norm_const
            ρmin, ρmax = extrema(ρk)
            normed = (ρk .- ρmin) ./ (ρmax - ρmin + 1e-8)

            c = colors[mod1(k, length(colors))]
            cmap = [Makie.RGBAf(1 - a*(1-c.r), 1 - a*(1-c.g), 1 - a*(1-c.b), 0.45*a^2) for a in range(0,1;length=256)]

            Makie.volume!(ax, extrema(X), extrema(Y), extrema(Z), normed;
                algorithm=:absorption, absorption=5f0, colormap=cmap)
        end
    end

    return fig
end

end
