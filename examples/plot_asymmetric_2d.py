#!/usr/bin/env python3
"""
plot_asymmetric_2d.py

Plot 2D asymmetric diblock copolymer SCFT results (cylinder phase).
Reads:
  asymmetric_2d.dat       — 2D density field data
  asymmetric_2d_conv.dat  — convergence history (optional)

Produces:
  asymmetric_2d.png  — 3-panel figure:
                          [ρ_A heatmap]  [ρ_B heatmap]  [convergence]
"""

import matplotlib
matplotlib.use("Agg")   # headless rendering (no display required)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load density data ─────────────────────────────────────────────────────────
datafile = os.path.join(SCRIPT_DIR, "asymmetric_2d_snapshot.dat")
if not os.path.exists(datafile):
    raise FileNotFoundError(
        f"Data file not found: {datafile}\n"
        "Run:  julia --project asymmetric_diblock_2d.jl"
    )

params_line = ""
with open(datafile) as f:
    for line in f:
        if line.startswith("# N="):
            params_line = line.strip("# \n")
            break

data = np.loadtxt(datafile)
ix_raw   = data[:, 0].astype(int)
iy_raw   = data[:, 1].astype(int)
x_flat   = data[:, 2]
y_flat   = data[:, 3]
rhoA_flat = data[:, 4]
rhoB_flat = data[:, 5]
# rhot_flat = data[:, 6]

# Determine grid dimensions
ngx = ix_raw.max()
ngy = iy_raw.max()
print(f"Grid: {ngx} × {ngy}")

# Reshape to 2D arrays (ix varies fastest in inner loop)
# Data written as:  for iy in 1:ngy, ix in 1:ngx
# So row-major order with iy as outer index
rhoA = rhoA_flat.reshape((ngy, ngx))
rhoB = rhoB_flat.reshape((ngy, ngx))
# rhot = rhot_flat.reshape((ngy, ngx))

Lx = x_flat.max()
Ly = y_flat.max()

print(f"ρ_A: [{rhoA.min():.4f}, {rhoA.max():.4f}]")
print(f"ρ_B: [{rhoB.min():.4f}, {rhoB.max():.4f}]")

# ── Load convergence log ──────────────────────────────────────────────────────
conv_file = os.path.join(SCRIPT_DIR, "asymmetric_2d_conv.dat")
has_conv  = os.path.exists(conv_file)
if has_conv:
    conv = np.loadtxt(conv_file)
    if conv.ndim == 1:
        conv = conv[np.newaxis, :]
    iters  = conv[:, 0]
    errors = conv[:, 1]
    fes    = conv[:, 2]

# ── Plot ──────────────────────────────────────────────────────────────────────
ncols = 3 if has_conv else 2
fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))

extent = [0, Lx, 0, Ly]

def add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

# Panel 1: ρ_A
ax = axes[0]
im = ax.imshow(rhoA, origin='lower', extent=extent, aspect='equal',
               cmap='Blues', interpolation='bilinear')
# Contour line at the mean to highlight the cylinder boundary
ax.contour(np.linspace(0, Lx, ngx), np.linspace(0, Ly, ngy),
           rhoA, levels=[rhoA.mean()], colors='white', linewidths=1.0)
add_colorbar(ax, im)
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$y$', fontsize=12)
ax.set_title(r'$\rho_A(x,y)$  [A-block density]', fontsize=12)
if params_line:
    ax.text(0.02, 0.98, params_line, transform=ax.transAxes, fontsize=6.5,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

# Panel 2: ρ_B
ax = axes[1]
im = ax.imshow(rhoB, origin='lower', extent=extent, aspect='equal',
               cmap='Reds', interpolation='bilinear')
ax.contour(np.linspace(0, Lx, ngx), np.linspace(0, Ly, ngy),
           rhoB, levels=[rhoB.mean()], colors='white', linewidths=1.0)
add_colorbar(ax, im)
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$y$', fontsize=12)
ax.set_title(r'$\rho_B(x,y)$  [B-block density]', fontsize=12)

# Panel 3 (optional): convergence
if has_conv:
    ax3 = axes[2]
    ax3.semilogy(iters, errors, color='steelblue', lw=2.0,
                 marker='o', markersize=4)
    ax3.axhline(1e-6, color='k', lw=0.8, ls='--', label='tol = 1e-6')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel(r'$\max|\,w_\mathrm{new} - w\,|$', fontsize=12)
    ax3.set_title('SCFT Convergence', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, which='both', ls=':', alpha=0.5)

    ax3b = ax3.twinx()
    ax3b.plot(iters, fes, color='darkorange', lw=1.5, ls='-', alpha=0.8)
    ax3b.set_ylabel('Free energy', color='darkorange', fontsize=10)
    ax3b.tick_params(axis='y', colors='darkorange')

fig.suptitle(r'2D Asymmetric Diblock  ($f_A=0.35$, $\chi N=35$)  — Cylinder Phase',
             fontsize=13, y=1.01)
fig.tight_layout()

outpath = os.path.join(SCRIPT_DIR, "asymmetric_2d.png")
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved → {outpath}")
plt.close()
