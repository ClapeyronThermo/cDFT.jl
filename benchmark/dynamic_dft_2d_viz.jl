using HDF5, Plots

densities, times = h5open("trajectory_test_2d_vle.h5", "r") do f
    rho_keys = sort(filter(k -> startswith(k, "rho_"), keys(f)),
                    by = k -> parse(Int, split(k, "_")[2]))
    t_keys   = sort(filter(k -> startswith(k, "t_"),   keys(f)),
                    by = k -> parse(Int, split(k, "_")[2]))

    densities = [read(f[rk])[:,:,:] for rk in rho_keys]
    times     = [read(f[tk])                for tk in t_keys]

    return densities, times
end

cols_w = [RGBA(c.r, c.g, c.b, α.^2)
        for (c,α) in zip(cgrad([:white,:blue],256),
                         range(0,1,length=256))]

cols_h = [RGBA(c.r, c.g, c.b, α.^2)
        for (c,α) in zip(cgrad([:white,:yellow],256),
                         range(0,1,length=256))]
cols_e = [RGBA(c.r, c.g, c.b, α.^2)
        for (c,α) in zip(cgrad([:white,:red],256),
                         range(0,1,length=256))]
anim = Animation()
anim = @animate for n=1:10:length(densities)
    # t=t+dt
    # if mod(n,40)==0
        # x = 0:0.1:2π
        # y = sin.(x .+ (n/2))
        z = densities[n]
        
        heatmap(
            z[:,:,1],
            c=cols_w,
            clim=(0, maximum(densities[end][:,:,1])),
            aspect_ratio=:equal,
            xaxis=false,
            yaxis=false,
            ticks=nothing,
            interpolate=true,
            colorbar=:right,
            size=(520, 400),
            margin=0Plots.mm,
            right_margin=12Plots.mm,
        )
        # heatmap!(z[:,:,2],c=cols_h, clim=(0,maximum(densities[end][:,:,2])), aspect_ratio=:equal, xaxis=false, yaxis=false, ticks=nothing, interpolate = true)
        # heatmap!(z[:,:,3],c=cols_e, clim=(0,maximum(densities[end][:,:,3])), aspect_ratio=:equal, xaxis=false, yaxis=false, ticks=nothing, interpolate = true)
        # heatmap!(z[:,:,4],c=cols_w, aspect_ratio=:equal, xaxis=false, yaxis=false, ticks=nothing)
        # heatmap!(z[:,:,3],c=cols_e, clim=(0,601.2641845581202), aspect_ratio=:equal, xaxis=false, yaxis=false, ticks=nothing)

        # xlims!(-20, 20)
        # ylims!(-20, 20)
        frame(anim)
    # end
end
gif(anim, "trajectory_2d_vle_hetero.gif", fps=30)