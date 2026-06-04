#!/usr/bin/env julia

using Plots
using Printf

function main()
    filename = "scan.txt"
    if !isfile(filename)
        println("Error: $filename not found.")
        return
    end

    ngrids = Float64[]
    old_cpu = Float64[]
    old_gpu = Float64[]
    new_cpu = Float64[]
    new_gpu = Float64[]

    lines = readlines(filename)
    for line in lines
        # Skip header and separator lines
        if occursin("NGRID", line) || occursin("---", line) || isempty(strip(line))
            continue
        end

        parts = split(line)
        if length(parts) >= 5
            try
                push!(ngrids, parse(Float64, parts[1]))
                push!(old_cpu, parse(Float64, parts[2]))
                push!(old_gpu, parse(Float64, parts[3]))
                push!(new_cpu, parse(Float64, parts[4]))
                push!(new_gpu, parse(Float64, parts[5]))
            catch e
                @warn "Could not parse line: $line"
            end
        end
    end

    if isempty(ngrids)
        println("No data found in $filename.")
        return
    end

    p = Plots.plot(
        xaxis=:log10, 
        yaxis=:log10, 
        xlabel="NGRID", 
        ylabel="Time (ms)", 
        title="cDFT(PC-SAFT) Benchmark Results (scan.jl)",
        legend=:topleft,
        marker=:circle,

        titlefontsize=16,
        guidefontsize=16,      # xlabel/ylabel
        tickfontsize=14,
        legendfontsize=13,
    )

    Plots.plot!(p, ngrids, old_cpu, label="old_cpu")
    Plots.plot!(p, ngrids, old_gpu, label="old_gpu")
    Plots.plot!(p, ngrids, new_cpu, label="new_cpu")
    Plots.plot!(p, ngrids, new_gpu, label="new_gpu")

    Plots.savefig(p, "scan_plot.png")
    println("Plot saved to scan_plot.png")
end

main()
