using HDF5, GLMakie, Colors

GLMakie.activate!()
# Read data
# rho, t = h5open("trajectory.h5", "r") do f
#     println("Available datasets: ", keys(f))
#     read(f["rho_17"]), read(f["t_17"])
# end

# rho_water  = rho[:, :, :, 1]
# rho_hexane = rho[:, :, :, 2]

# # Volume plot
# fig, ax, plt = volume(rho_water,
#     algorithm  = :iso,
#     isorange   = 5.0,
#     isovalue   = sum(rho_water)/length(rho_water),
#     colormap   = :RdBu,
#     colorrange = (minimum(rho_water), maximum(rho_water)),
#     axis       = (type=Axis3, title="Water Density, t=$t")
# )

# save("water_density.png", fig)

# display(fig)

ngrid = 101

function interpolate((x,), y, method)
    
end

function animate_trajectory(h5path, species=1; save_every=1, framerate=10)
    # Read all timestep keys
    densities, times = h5open(h5path, "r") do f
        rho_keys = sort(filter(k -> startswith(k, "rho_"), keys(f)),
                        by = k -> parse(Int, split(k, "_")[2]))
        t_keys   = sort(filter(k -> startswith(k, "t_"),   keys(f)),
                        by = k -> parse(Int, split(k, "_")[2]))

        densities = [read(f[rk])[:,:,:,species] for rk in rho_keys]
        times     = [read(f[tk])                for tk in t_keys]

        # Linearly interpolate so that the densities have a time step of 1/save_every between frames
        t_min, t_max = minimum(times), maximum(times)
        t_interp = range(t_min, t_max, step=1/save_every)
        densities_interp = Vector{Array{eltype(densities[1]),4}}()
        for i in eachindex(t_interp)
            if i == 1
                densities_interp = [densities[i]]
            elseif i == length(t_interp)
                push!(densities_interp, densities[end])
            else
                # Find the two closest time points
                t_prev = maximum(filter(t -> t <= t_interp[i], times))
                t_next = minimum(filter(t -> t >= t_interp[i], times))
                ρ_prev = densities[findfirst(==(t_prev), times)]
                ρ_next = densities[findfirst(==(t_next), times)]

                # Linear interpolation
                α = (t_interp[i] - t_prev) / (t_next - t_prev + 1e-8)
                push!(densities_interp, (1-α) * ρ_prev + α * ρ_next)
            end
        end

        return densities_interp, t_interp
    end

    # Observable for current frame
    frame_idx = Observable(1)
    ρ_current = @lift(densities[$frame_idx])
    t_current = @lift(times[$frame_idx])

    # Build figure
    fig = Figure(size=(800, 800))
    ax  = Axis3(fig[1,1],
        aspect  = (ngrid, ngrid, ngrid),   # correct aspect ratio from actual dims
        viewmode    = :fit,
        # limits      = ((1, ngrid), (1, ngrid), (1, ngrid)),   # fixed axis limits
        # azimuth   = π/4,      # fixed rotation angle
        # elevation = π/6,      # fixed elevation angle
        # perspectiveness = 0.5,    # 0=orthographic, 1=full perspective
        limits          = (
            (1, ngrid),       # zoom into central region
            (1, ngrid),
            (1, ngrid)
        ),
        xspinesvisible  = false,
        yspinesvisible  = false,
        zspinesvisible  = false,
        xgridvisible    = false,
        ygridvisible    = false,
        zgridvisible    = false,
        xticksvisible   = false,
        yticksvisible   = false,
        zticksvisible   = false,
        xlabelvisible   = false,
        ylabelvisible   = false,
        zlabelvisible   = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        zticklabelsvisible = false,
    )
    
    # cam = cameracontrols(ax.scene)
    # update_cam!(ax.scene, cam,
    #     Vec3f(ngrid/2, -ngrid, ngrid/2),    # eye position
    #     Vec3f(ngrid/2, ngrid/2, ngrid/2),   # look-at point
    #     Vec3f(0, 0, 1)              # up vector
    # )
    clims = (minimum(densities[end]), maximum(densities[end]))
    cols_w = [RGBA(c.r, c.g, c.b, α^2)
        for (c,α) in zip(cgrad([:white,:red],256),
                         range(0,1,length=256))]

    normed = @lift(($ρ_current .- $clims[1]) ./ ($clims[2] - $clims[1] + 1e-8))
    volume!(ax, 1 .. ngrid, 1 .. ngrid, 1 .. ngrid,  normed,
        algorithm  = :absorption,
        absorption = 5,
        colormap   = cols_w,
    )
    
    cam3d!(ax.scene)


    # Label(fig[1,1], @lift("t = $(round($t_current, digits=3))"),
        #   fontsize=24, padding=(10,10))
    

    nframes = min(10, length(densities))
    frame_ids = round.(Int, range(1, length(densities), length=nframes))
    
    # Animate
    GLMakie.record(fig, "trajectory_test_3d.gif", frame_ids; framerate) do i
        frame_idx[] = i
        # ax.azimuth[]   = π/4    # reset to fixed values each frame
        # ax.elevation[] = π/6
        # reset_limits!(ax)
    end
end

animate_trajectory("trajectory_test_3d.h5", 1)