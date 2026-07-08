module MakiecDFTExt

using cDFT
using Makie

_maybe_texlabel(s, latex::Bool) = latex ? cDFT.texlabel(s) : s

# `segment`/`size` convert a number-density profile to a dimensionless volume fraction
# for segment-based (SAFT-style) models. SCFT's `SCFTLatticeFluid` has no such params —
# its converged profiles are already volume fractions — so those models get norm_const=1.
function _norm_const(species, model, i::Int, k::Int)
    hasproperty(model.params, :segment) || return 1.0
    return species.nbeads[i] > 1 ? model.params.segment[k]*species.size[k]^3*cDFT.N_A : model.params.segment[i]*species.size[i]^3*cDFT.N_A
end

_normalized_ylabel(model) = hasproperty(model.params, :segment) ? "ρσ³" : "φ"

# ── Aggregation ("plot_by") / coloring ("color_by") ──────────────────────
#
# Both are ∈ (:bead, :group, :molecule), from finest to coarsest granularity. `:bead`
# matches today's default (one curve per flattened bead index, e.g. "A_1"/"A_2"/"B_1");
# `:group` averages instances of the same named group within a component together (e.g.
# all "A_*" beads into one "A" curve); `:molecule` averages every bead of a component into
# one curve. `color_by` controls color assignment only, independent of aggregation, but
# can't be finer than `plot_by` (no per-bead data survives once beads are averaged
# away).

const _LEVEL_RANK = (bead = 1, group = 2, molecule = 3)

function _check_profile_color_by(plot_by::Symbol, color_by::Symbol)
    haskey(_LEVEL_RANK, plot_by) || error("plot_by must be :bead, :group or :molecule, got :$plot_by")
    haskey(_LEVEL_RANK, color_by) || error("color_by must be :bead, :group or :molecule, got :$color_by")
    _LEVEL_RANK[color_by] >= _LEVEL_RANK[plot_by] || error(
        "color_by=:$color_by cannot be finer-grained than plot_by=:$plot_by " *
        "(granularity order: bead < group < molecule) — there's no per-$(color_by) data " *
        "left once beads have been averaged to the :$plot_by level.")
end

function _group_key(species, model, i::Int, k::Int, level::Symbol)
    if level === :molecule
        return (i,)
    elseif level === :group
        return species.nbeads[i] > 1 ? (i, cDFT._group_letter(model.groups.flattenedgroups[k])) : (i,)
    else # :bead
        return (i, k)
    end
end

function _profile_label(species, model, i::Int, k::Int, level::Symbol)
    species_name = model.components[i]
    if level === :molecule || species.nbeads[i] == 1
        return species_name
    elseif level === :group
        return "$species_name $(cDFT._group_letter(model.groups.flattenedgroups[k]))"
    else # :bead
        return "$species_name $(model.groups.flattenedgroups[k])"
    end
end

# One entry per curve/field to draw: (label, [(i,k) beads to average together]).
function _plot_groups(species, model, plot_by::Symbol)
    members = Dict{Any,Vector{Tuple{Int,Int}}}()
    order = Any[]
    for i in cDFT.@comps
        for k in cDFT.@chain(i)
            key = _group_key(species, model, i, k, plot_by)
            if !haskey(members, key)
                members[key] = Tuple{Int,Int}[]
                push!(order, key)
            end
            push!(members[key], (i, k))
        end
    end
    return [(_profile_label(species, model, members[key][1]..., plot_by), members[key]) for key in order]
end

# Dict{color_key,color}, assigned in first-encountered order from Makie.wong_colors().
function _assign_colors(color_keys)
    palette = Makie.wong_colors()
    colors = Dict{Any,Any}()
    idx = 0
    for key in color_keys
        if !haskey(colors, key)
            idx += 1
            colors[key] = palette[mod1(idx, length(palette))]
        end
    end
    return colors
end

# For each (label, members) group from `_plot_groups`, its assigned color (keyed at
# `color_by` granularity, which may be coarser than `plot_by` so several groups can
# share one color).
function _group_colors(groups, species, model, color_by::Symbol)
    color_keys = [_group_key(species, model, members[1]..., color_by) for (_, members) in groups]
    colors = _assign_colors(color_keys)
    return [colors[key] for key in color_keys]
end

function Makie.plot(system::cDFT.AbstractcDFTSystem, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead)
    return Makie.plot(system, system.structure, profiles; x_units=x_units, y_units=y_units, latex=latex, plot_by=plot_by, color_by=color_by)
end

function Makie.plot(system::cDFT.AbstractcDFTSystem, structure::cDFT.DFTStructure1DCart, profiles; x_units=:normalized, y_units=:mass, latex=false, plot_by=:bead, color_by=:bead)
    _check_profile_color_by(plot_by, color_by)
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

    if x_units == :normalized
        X = z./L
    elseif x_units == :angstrom
        X = z.*1e10
    elseif x_units == :nanometer
        X = z.*1e9
    else
        X = z
    end

    function bead_Y(i, k)
        norm_const = _norm_const(species, model, i, k)
        if y_units == :normalized
            return profiles[:,k].*norm_const
        elseif y_units == :mass
            Mw = model.params.Mw[k]
            return profiles[:,k].*Mw/1e3
        elseif y_units == :angstrom
            return profiles[:,k].*cDFT.N_A/1e30
        else
            return profiles[:,k]
        end
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by)

    ymax = 0.
    for ((label, members), c) in zip(groups, colors)
        Y = sum(bead_Y(i,k) for (i,k) in members) ./ length(members)
        Makie.lines!(ax, X, Y; label=_maybe_texlabel(label,latex), linewidth=3, color=c)
        ymax = max(ymax,maximum(Y))
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
        ax.ylabel = _maybe_texlabel(_normalized_ylabel(model),latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt)

    return fig
end

function Makie.plot(system::cDFT.AbstractcDFTSystem, structure::Union{cDFT.DFTStructure1DSphr,cDFT.DFTStructure1DCyl}, profiles; x_units=:normalized, y_units=:mass, latex=false, plot_by=:bead, color_by=:bead)
    _check_profile_color_by(plot_by, color_by)
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

    if x_units == :normalized
        X = z./L
    elseif x_units == :angstrom
        X = z.*1e10
    elseif x_units == :nanometer
        X = z.*1e9
    else
        X = z
    end

    function bead_Y(i, k)
        norm_const = _norm_const(species, model, i, k)
        if y_units == :normalized
            return profiles[:,k].*norm_const
        elseif y_units == :mass
            Mw = model.params.Mw[k]
            return profiles[:,k].*Mw/1e3
        elseif y_units == :angstrom
            return profiles[:,k].*cDFT.N_A/1e30
        else
            return profiles[:,k]
        end
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by)

    ymax = 0.
    for ((label, members), c) in zip(groups, colors)
        Y = sum(bead_Y(i,k) for (i,k) in members) ./ length(members)
        Makie.lines!(ax, X, Y; label=_maybe_texlabel(label,latex), linewidth=3, color=c)
        ymax = max(ymax,maximum(Y))
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
        ax.ylabel = _maybe_texlabel(_normalized_ylabel(model),latex)
    elseif y_units == :mass
        ax.ylabel = _maybe_texlabel("ρ / (kg/m³)",latex)
    else
        ax.ylabel = _maybe_texlabel("ρ / (mol/m³)",latex)
    end

    Makie.axislegend(ax; position=:lt)

    return fig
end

function Makie.plot(system::Union{cDFT.DFTSystem,cDFT.DGTSystem}, structure::cDFT.DFTStructure2DCart, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead)
    _check_profile_color_by(plot_by, color_by)
    structure = system.structure
    model = system.model
    species = system.species

    bounds = structure.bounds

    x = cDFT.uniform_range(structure,1)
    y = cDFT.uniform_range(structure,2)
    L = cDFT.length_scale(model)

    fig = Figure()
    ax = Axis(fig[1, 1]; xgridvisible=false, ygridvisible=false, aspect=Makie.DataAspect())

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
    elseif y_units == :angstrom
        Y = y.*1e10
    elseif y_units == :nanometer
        Y = y.*1e9
    else
        Y = y
    end

    function bead_Z(i, k)
        norm_const = _norm_const(species, model, i, k)
        return profiles[:,:,k].*norm_const
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by)

    for ((label, members), c) in zip(groups, colors)
        Z = sum(bead_Z(i,k) for (i,k) in members) ./ length(members)
        csalpha = [Makie.RGBAf(c.r, c.g, c.b, 0.0), Makie.RGBAf(c.r, c.g, c.b, 1.0)]
        Makie.heatmap!(ax, X, Y, Z; colormap=csalpha, label=_maybe_texlabel(label,latex))
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

function Makie.plot(system::Union{cDFT.DFTSystem,cDFT.DGTSystem}, structure::cDFT.DFTStructure3DCart, profiles; x_units=:normalized, y_units=:normalized, latex=false, plot_by=:bead, color_by=:bead)
    _check_profile_color_by(plot_by, color_by)
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

    function bead_ρ(i, k)
        norm_const = _norm_const(species, model, i, k)
        return profiles[:,:,:,k].*norm_const
    end

    groups = _plot_groups(species, model, plot_by)
    colors = _group_colors(groups, species, model, color_by)

    for ((label, members), c) in zip(groups, colors)
        ρk = sum(bead_ρ(i,k) for (i,k) in members) ./ length(members)
        ρmin, ρmax = extrema(ρk)
        normed = (ρk .- ρmin) ./ (ρmax - ρmin + 1e-8)

        cmap = [Makie.RGBAf(c.r, c.g, c.b, 0.45*a^2) for a in range(0,1;length=256)]

        Makie.volume!(ax, extrema(X), extrema(Y), extrema(Z), normed;
            algorithm=:absorption, absorption=5f0, colormap=cmap)
    end

    return fig
end

end
