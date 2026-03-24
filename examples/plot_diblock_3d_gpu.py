#!/usr/bin/env python3
"""
plot_diblock_3d_gpu.py

Plot 3D diblock copolymer solution SCFT results (GPU run).
Reads:
  diblock_solution_3d_gpu.dat       — volumetric density data
  diblock_solution_3d_gpu_conv.dat  — convergence history (optional)

Produces:
  diblock_solution_3d_gpu_isosurface.png  — Plotly isosurface (A, B, solvent)
  diblock_solution_3d_gpu_slices.png      — matplotlib cross-section slices
  diblock_solution_3d_gpu_conv.png        — convergence curve

Requires: plotly, kaleido  (pip install plotly kaleido)
          numpy, matplotlib (standard)
"""

import matplotlib
matplotlib.use("Agg")   # headless rendering (no display required)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load density data ─────────────────────────────────────────────────────────
datafile = os.path.join(SCRIPT_DIR, "diblock_solution_3d_gpu_snapshot.dat")
if not os.path.exists(datafile):
    raise FileNotFoundError(
        f"Data file not found: {datafile}\n"
        "Run:  julia --project diblock_solution_3d_gpu.jl"
    )

meta = {}
with open(datafile) as f:
    for line in f:
        if not line.startswith("#"):
            break
        for key in ("N=", "chi_AB=", "chiN=", "phi_polymer=", "phi_solvent=",
                    "ngrid=", "L="):
            if key in line:
                for tok in line.strip("# \n").split(","):
                    tok = tok.strip()
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k.strip()] = v.strip()

data = np.loadtxt(datafile)
ix  = data[:, 0].astype(int)
iy  = data[:, 1].astype(int)
iz  = data[:, 2].astype(int)
x   = data[:, 3]
y   = data[:, 4]
z   = data[:, 5]
rA  = data[:, 6]
rB  = data[:, 7]
rS  = data[:, 8]
rT  = data[:, 9]

ngrid = int(ix.max())
L     = float(x.max())
print(f"Grid: {ngrid}³ = {ngrid**3} points, L={L}")

# Reshape: written as  for iz in 1:ng, iy in 1:ng, ix in 1:ng
# So ix is fastest-varying → C-order with dims (nz, ny, nx)
RA = rA.reshape((ngrid, ngrid, ngrid))   # (iz, iy, ix) indexing
RB = rB.reshape((ngrid, ngrid, ngrid))
RS = rS.reshape((ngrid, ngrid, ngrid))
RT = rT.reshape((ngrid, ngrid, ngrid))

coords = np.linspace(0, L, ngrid)

print(f"ρ_A: [{RA.min():.4f}, {RA.max():.4f}]")
print(f"ρ_B: [{RB.min():.4f}, {RB.max():.4f}]")
print(f"ρ_S: [{RS.min():.4f}, {RS.max():.4f}]")
print(f"ρ_total: [{RT.min():.5f}, {RT.max():.5f}]")

# ── 1. Cross-section slices (matplotlib) ─────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

iz_mid = ngrid // 2   # mid-plane slice
iy_mid = ngrid // 2

slice_titles = [
    (r'$\rho_A$', RA, 'Blues'),
    (r'$\rho_B$', RB, 'Reds'),
    (r'$\rho_S$ (solvent)', RS, 'Greens'),
]

for col, (title, R, cmap) in enumerate(slice_titles):
    # z-slice (x-y plane at z=L/2)
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(R[iz_mid, :, :], origin='lower',
                   extent=[0, L, 0, L], aspect='equal',
                   cmap=cmap, interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{title}  (z = {L/2:.1f})', fontsize=11)
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$y$', fontsize=10)

    # x-slice (y-z plane at x=L/2) — shows lamellar structure along z
    ax2 = fig.add_subplot(gs[1, col])
    im2 = ax2.imshow(R[:, iy_mid, :], origin='lower',
                     extent=[0, L, 0, L], aspect='equal',
                     cmap=cmap, interpolation='bilinear')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title(f'{title}  (y = {L/2:.1f})', fontsize=11)
    ax2.set_xlabel(r'$x$', fontsize=10)
    ax2.set_ylabel(r'$z$', fontsize=10)

chi_AB = meta.get("chi_AB", "1.5")
chiN   = meta.get("chiN", "30")
fig.suptitle(
    f'3D Diblock Copolymer Solution  '
    r'($\chi_{AB}$' + f'={chi_AB}, ' + r'$\chi N$' + f'={chiN})  —  Lamellar Phase',
    fontsize=13, y=1.01
)
outpath_slices = os.path.join(SCRIPT_DIR, "diblock_solution_3d_gpu_slices.png")
fig.savefig(outpath_slices, dpi=150, bbox_inches='tight')
print(f"Saved → {outpath_slices}")
plt.close()

# ── 2. Isosurface plots (Plotly, one figure per species) ──────────────────────
try:
    import plotly.graph_objects as go
    import gc

    # Build flat coordinate arrays from reshaped grid, then subsample.
    # RA/RB/RS are (nz, ny, nx); transpose to (nx, ny, nz) for x-major order.
    RA_xyz = np.transpose(RA, (2, 1, 0))
    RB_xyz = np.transpose(RB, (2, 1, 0))
    RS_xyz = np.transpose(RS, (2, 1, 0))

    Xg, Yg, Zg = np.meshgrid(coords, coords, coords, indexing='ij')

    stride = max(1, ngrid // 50)
    sl = slice(None, None, stride)
    xf = Xg[sl, sl, sl].flatten()
    yf = Yg[sl, sl, sl].flatten()
    zf = Zg[sl, sl, sl].flatten()
    print(f"Isosurface grid after subsampling (stride={stride}): {len(xf)} points")

    species = [
        ('rho_A', RA_xyz, 'Blues'),
        ('rho_B', RB_xyz, 'Reds'),
        ('rho_S', RS_xyz, 'Greens'),
    ]

    iso_outpaths = []
    for name, R_xyz, cscale in species:
        vf = R_xyz[sl, sl, sl].flatten()
        print(f"{name}: mean={vf.mean():.4f}, max={vf.max():.4f}")

        vol = go.Volume(
            x=xf, y=yf, z=zf,
            value=vf,
            isomin=vf.min() + 0.4 * (vf.max() - vf.min()),
            isomax=vf.max() - 0.4 * (vf.max() - vf.min()),
            surface_count=8,
            opacity=0.4,
            colorscale=cscale,
            flatshading=False,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

        layout = go.Layout(
            autosize=False, width=700, height=500,
            margin=go.layout.Margin(l=20, r=20, b=20, t=40, pad=0),
            title=dict(
                text=f'3D Diblock Solution (χ_AB={chi_AB}, χN={chiN}) — {name}',
                x=0.5, font=dict(size=12),
            ),
        )

        fig3d = go.Figure(data=[vol], layout=layout)
        fig3d.update_scenes(camera_projection_type='orthographic')
        fig3d.update_layout(
            coloraxis={'colorscale': cscale},
            scene=dict(aspectmode='data'),
        )

        outpath = os.path.join(SCRIPT_DIR, f"diblock_solution_3d_gpu_isosurface_{name}.png")
        fig3d.write_image(outpath, scale=2)
        print(f"Saved → {outpath}")
        iso_outpaths.append(outpath)
        plt.close('all')
        gc.collect()

    # Keep the primary output name pointing at rho_A for backwards compatibility.
    import shutil
    if iso_outpaths:
        shutil.copy(iso_outpaths[0],
                    os.path.join(SCRIPT_DIR, "diblock_solution_3d_gpu_isosurface.png"))

except ImportError:
    print("Plotly/kaleido not available — skipping isosurface plot.")
    print("Install with:  pip install plotly kaleido")

# ── 3. Convergence plot ───────────────────────────────────────────────────────
conv_file = os.path.join(SCRIPT_DIR, "diblock_solution_3d_gpu_conv.dat")
if os.path.exists(conv_file):
    conv = np.loadtxt(conv_file)
    if conv.ndim == 1:
        conv = conv[np.newaxis, :]
    iters  = conv[:, 0]
    errors = conv[:, 1]
    fes    = conv[:, 2]

    fig2, ax1 = plt.subplots(figsize=(7, 4))
    ax1.semilogy(iters, errors, color='steelblue', lw=2.0,
                 marker='o', markersize=4, label='field residual')
    ax1.axhline(1e-6, color='k', lw=0.8, ls='--', label='tol = 1e-6')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel(r'$\max|\,w_\mathrm{new} - w\,|$', fontsize=12)
    ax1.set_title(f'3D GPU SCFT Convergence  (χN={chiN})', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(iters, fes, color='tomato', lw=1.5, ls='-', alpha=0.8)
    ax2.set_ylabel('Free energy', color='tomato', fontsize=10)
    ax2.tick_params(axis='y', colors='tomato')

    fig2.tight_layout()
    outpath_conv = os.path.join(SCRIPT_DIR, "diblock_solution_3d_gpu_conv.png")
    fig2.savefig(outpath_conv, dpi=150, bbox_inches='tight')
    print(f"Saved → {outpath_conv}")
    plt.close()
