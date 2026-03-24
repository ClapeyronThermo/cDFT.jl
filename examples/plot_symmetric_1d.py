#!/usr/bin/env python3
"""
plot_symmetric_1d.py

Plot 1D symmetric diblock copolymer SCFT results.
Reads:
  symmetric_1d.dat       — density profiles
  symmetric_1d_conv.dat  — convergence history (optional)

Produces:
  symmetric_1d.png  — 2-panel figure: density profiles + error convergence
"""

import matplotlib
matplotlib.use("Agg")   # headless rendering (no display required)
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load density profiles ────────────────────────────────────────────────────
datafile = os.path.join(SCRIPT_DIR, "symmetric_1d.dat")
if not os.path.exists(datafile):
    raise FileNotFoundError(
        f"Data file not found: {datafile}\n"
        "Run:  julia --project symmetric_diblock_1d.jl"
    )

# Parse header for parameter labels
params_line = ""
fe_line = ""
with open(datafile) as f:
    for line in f:
        if line.startswith("# N=") and not params_line:
            params_line = line.strip("# \n")
        if "Free energy" in line:
            fe_line = line.strip("# \n")

data     = np.loadtxt(datafile)
z        = data[:, 0]
rho_A    = data[:, 1]
rho_B    = data[:, 2]
rho_tot  = data[:, 3]

# ── Load convergence log (optional) ─────────────────────────────────────────
conv_file = os.path.join(SCRIPT_DIR, "symmetric_1d_conv.dat")
has_conv = os.path.exists(conv_file)
if has_conv:
    conv = np.loadtxt(conv_file)
    if conv.ndim == 1:
        conv = conv[np.newaxis, :]
    iters  = conv[:, 0]
    errors = conv[:, 1]
    fes    = conv[:, 2]

# ── Plot ─────────────────────────────────────────────────────────────────────
ncols = 2 if has_conv else 1
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
if ncols == 1:
    axes = [axes]

# Panel 1: density profiles
ax = axes[0]
ax.plot(z, rho_A,   color='royalblue', lw=2.0, label=r'$\rho_A$')
ax.plot(z, rho_B,   color='tomato',    lw=2.0, label=r'$\rho_B$')
ax.plot(z, rho_tot, color='k',         lw=1.2, ls='--', label=r'$\rho_\mathrm{total}$')

ax.axhline(1.0, color='gray', lw=0.8, ls=':', label=r'$\rho_0 = 1$')
ax.set_xlabel(r'$z$', fontsize=13)
ax.set_ylabel(r'$\rho(z)$', fontsize=13)
ax.set_title('Lamellar Density Profiles\n'
             r'Symmetric AB, $\chi N = 30$', fontsize=12)
ax.set_xlim(z[0], z[-1])
ax.set_ylim(0, None)
ax.legend(fontsize=11)
if params_line:
    ax.text(0.02, 0.97, params_line, transform=ax.transAxes, fontsize=7,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

# Panel 2: error convergence
if has_conv:
    ax2 = axes[1]
    ax2.semilogy(iters, errors, color='royalblue', lw=2.0, marker='o',
                 markersize=4, label='field residual')
    ax2.axhline(1e-6, color='k', lw=0.8, ls='--', label='tol = 1e-6')
    ax2.set_xlabel('Iteration', fontsize=13)
    ax2.set_ylabel(r'$\max|\,w_\mathrm{new} - w\,|$', fontsize=13)
    ax2.set_title('SCFT Picard Convergence', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, which='both', ls=':', alpha=0.5)

    # Add free-energy inset
    ax2b = ax2.twinx()
    ax2b.plot(iters, fes, color='tomato', lw=1.5, ls='-', alpha=0.7)
    ax2b.set_ylabel(r'$F / n_\mathrm{chains}$', color='tomato', fontsize=11)
    ax2b.tick_params(axis='y', colors='tomato')

fig.tight_layout()
outpath = os.path.join(SCRIPT_DIR, "symmetric_1d.png")
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved → {outpath}")
plt.close()
