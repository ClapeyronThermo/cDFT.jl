#!/usr/bin/env python3
"""
Plot 3D volumetric density profiles from diblock_solution_3d.dat using Plotly.
Renders isosurfaces of ρ_A and ρ_B to visualize microphase separation.
Saves output as PNG.
"""

import numpy as np
import os

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Install plotly: pip install plotly kaleido")

datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diblock_solution_3d.dat")

# Read metadata
params = {}
with open(datafile) as f:
    for line in f:
        if not line.startswith("#"):
            break

# Load data
data = np.loadtxt(datafile)
ix = data[:, 0].astype(int)
iy = data[:, 1].astype(int)
iz = data[:, 2].astype(int)
x = data[:, 3]
y = data[:, 4]
z = data[:, 5]
rho_A = data[:, 6]
rho_B = data[:, 7]
rho_S = data[:, 8]

# Determine grid dimensions
ngrid = int(round(max(ix)))
print(f"Grid: {ngrid}x{ngrid}x{ngrid} = {ngrid**3} points")

# Reshape to 3D arrays
X = x.reshape((ngrid, ngrid, ngrid), order='F')
Y = y.reshape((ngrid, ngrid, ngrid), order='F')
Z = z.reshape((ngrid, ngrid, ngrid), order='F')
RA = rho_A.reshape((ngrid, ngrid, ngrid), order='F')
RB = rho_B.reshape((ngrid, ngrid, ngrid), order='F')
RS = rho_S.reshape((ngrid, ngrid, ngrid), order='F')

print(f"ρ_A range: [{RA.min():.4f}, {RA.max():.4f}]")
print(f"ρ_B range: [{RB.min():.4f}, {RB.max():.4f}]")
print(f"ρ_S range: [{RS.min():.4f}, {RS.max():.4f}]")

# Compute isosurface thresholds
rho_A_mean = RA.mean()
rho_B_mean = RB.mean()
rho_A_amp = RA.max() - RA.min()
rho_B_amp = RB.max() - RB.min()

# Use thresholds above the mean to show enriched regions
iso_A = rho_A_mean + 0.3 * rho_A_amp
iso_B = rho_B_mean + 0.3 * rho_B_amp

print(f"Isosurface thresholds: A={iso_A:.4f}, B={iso_B:.4f}")

fig = go.Figure()

# A-block isosurface (blue)
fig.add_trace(go.Isosurface(
    x=X.flatten(order='F'),
    y=Y.flatten(order='F'),
    z=Z.flatten(order='F'),
    value=RA.flatten(order='F'),
    isomin=iso_A,
    isomax=RA.max(),
    surface_count=2,
    colorscale=[[0, 'rgba(0,0,255,0.3)'], [1, 'rgba(0,0,255,0.8)']],
    showscale=False,
    name='ρ_A',
    caps=dict(x_show=False, y_show=False, z_show=False),
))

# B-block isosurface (red)
fig.add_trace(go.Isosurface(
    x=X.flatten(order='F'),
    y=Y.flatten(order='F'),
    z=Z.flatten(order='F'),
    value=RB.flatten(order='F'),
    isomin=iso_B,
    isomax=RB.max(),
    surface_count=2,
    colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.8)']],
    showscale=False,
    name='ρ_B',
    caps=dict(x_show=False, y_show=False, z_show=False),
))

fig.update_layout(
    title=dict(
        text='3D Diblock Copolymer Solution (χ_AB=1.0, χ_AS=2.0, χ_BS=0.3)',
        x=0.5,
    ),
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube',
    ),
    legend=dict(x=0.02, y=0.98),
    width=900,
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diblock_solution_3d.png")
fig.write_image(outpath, scale=2)
print(f"Saved plot to {outpath}")
