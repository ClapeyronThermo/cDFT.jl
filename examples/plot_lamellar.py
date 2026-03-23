#!/usr/bin/env python3
"""Plot SCFT lamellar density profiles from lamellar_scft.dat"""

import numpy as np
import matplotlib.pyplot as plt
import os

datafile = os.path.join(os.path.dirname(__file__), "lamellar_scft.dat")
data = np.loadtxt(datafile)

z = data[:, 0]
rho_A = data[:, 1]
rho_B = data[:, 2]
rho_total = data[:, 3]

# Read metadata from header
with open(datafile) as f:
    for line in f:
        if line.startswith("# chi_AB"):
            params = line.strip()
        elif line.startswith("# Free energy"):
            fe_line = line.strip()
        elif not line.startswith("#"):
            break

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z, rho_A, 'b-', linewidth=2, label=r'$\rho_A$')
ax.plot(z, rho_B, 'r-', linewidth=2, label=r'$\rho_B$')
ax.plot(z, rho_total, 'k--', linewidth=1.5, label=r'$\rho_{\mathrm{total}}$')

ax.set_xlabel(r'$z$', fontsize=14)
ax.set_ylabel(r'$\rho(z)$', fontsize=14)
ax.set_title(r'SCFT Lamellar Density Profiles ($\chi N = 30$)', fontsize=14)
ax.legend(fontsize=12)
ax.set_xlim(z[0], z[-1])
ax.set_ylim(0, None)

# Add parameter annotation
ax.text(0.02, 0.97, params.lstrip('# '), transform=ax.transAxes,
        fontsize=8, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), "lamellar_scft.png")
fig.savefig(outpath, dpi=150)
print(f"Saved plot to {outpath}")
plt.close()
