#!/usr/bin/env python3
"""
plot_benchmarks.py

Publication-quality figures for SCFT backend benchmark results.

Backends compared:
  - CPU (Apple M4 Max, Float64, 4 threads)
  - CPU (Intel i9-10850K, Float64, 4 threads)
  - CUDA Float32 (NVIDIA RTX 3080 Ti)
  - CUDA Float64 (NVIDIA RTX 3080 Ti)
  - Metal Float32 (Apple M4 Max)

System types:
  - Polymer: symmetric AB diblock copolymer with chain propagators (N=30)
  - Solvent: single-species grand-canonical monomeric solvent (no propagators)

Figures generated:
  fig1_polymer_scaling.pdf    — log-log time vs grid size, polymer, all backends
  fig2_solvent_scaling.pdf    — log-log time vs grid size, solvent, all backends
  fig3_speedup.pdf            — GPU/M4 speedup over Intel i9, both systems
  fig4_cuda_precision.pdf     — CUDA Float32 vs Float64 across dimensions
  fig5_throughput.pdf         — grid-point throughput (Mpts/s), polymer
  fig6_summary_bars.pdf       — backend comparison at representative grid sizes
  fig7_propagator_overhead.pdf — polymer / solvent time ratio (chain propagation cost)
  fig8_combined_overview.pdf  — 2×3 combined overview of all backends & systems

Usage:
    python benchmarks/plot_benchmarks.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Colour scheme via Palettable ──────────────────────────────────────────────
try:
    import palettable
    _tab = palettable.cartocolors.qualitative.Prism_5.mpl_colors
    print("Using Palettable / CartoCOlors Prism_5 colour scheme.")
except Exception:
    print("Palettable not found; falling back to manual Prism hex codes.")
    _hex = ['#5F4690', '#1D6996', '#38A6A5', '#0F8554', '#73AF48']
    _tab = [matplotlib.colors.to_rgb(h) for h in _hex]

# One colour + linestyle + marker per backend
STYLE = {
    'cpu_m4':   dict(label='CPU (M4 Max, F64)',       color=_tab[0], ls='-',  marker='o', ms=5, zorder=6),
    'cpu_i9':   dict(label='CPU (i9-10850K, F64)',    color=_tab[1], ls='-',  marker='s', ms=5, zorder=5),
    'cuda_f32': dict(label='CUDA F32 (RTX 3080 Ti)',  color=_tab[2], ls='--', marker='^', ms=6, zorder=4),
    'cuda_f64': dict(label='CUDA F64 (RTX 3080 Ti)',  color=_tab[3], ls='--', marker='v', ms=6, zorder=4),
    'metal':    dict(label='Metal (M4 Max, F32)',      color=_tab[4], ls=':',  marker='D', ms=5, zorder=4),
}

# ── Matplotlib global style ───────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':             9,
    'axes.titlesize':       10,
    'axes.labelsize':        9,
    'xtick.labelsize':       8,
    'ytick.labelsize':       8,
    'legend.fontsize':       8,
    'legend.framealpha':     0.92,
    'legend.edgecolor':      '0.75',
    'legend.borderpad':      0.5,
    'axes.linewidth':        0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':             True,
    'grid.color':            '0.88',
    'grid.linewidth':        0.5,
    'grid.linestyle':        '-',
    'lines.linewidth':       1.6,
    'lines.markersize':      5,
    'errorbar.capsize':      2.5,
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':    0.05,
    'pdf.fonttype':          42,    # embed fonts as TrueType
    'ps.fonttype':           42,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.minor.visible':   True,
    'ytick.minor.visible':   True,
})

# ── Data loading ──────────────────────────────────────────────────────────────

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

FILE_MAP = {
    ('cpu_m4',   'polymer'): 'bench_cpu_M4-Max_polymer_results.csv',
    ('cpu_m4',   'solvent'): 'bench_cpu_M4-Max_solvent_results.csv',
    ('cpu_i9',   'polymer'): 'bench_cpu_i9-10850k_polymer_results.csv',
    ('cpu_i9',   'solvent'): 'bench_cpu_i9-10850k_solvent_results.csv',
    ('cuda_f32', 'polymer'): 'bench_cuda_3080Ti_f32_polymer_results.csv',
    ('cuda_f32', 'solvent'): 'bench_cuda_3080Ti_f32_solvent_results.csv',
    ('cuda_f64', 'polymer'): 'bench_cuda_3080Ti_f64_polymer_results.csv',
    ('cuda_f64', 'solvent'): 'bench_cuda_3080Ti_f64_solvent_results.csv',
    ('metal',    'polymer'): 'bench_metal_M4-Max_polymer_results.csv',
    ('metal',    'solvent'): 'bench_metal_M4-Max_solvent_results.csv',
}


def _load_csv(path):
    df = pd.read_csv(path, comment='#', header=None,
                     names=['dim', 'ng', 'n_pts', 't_ms', 't_std_ms'])
    df['t_ms']     = pd.to_numeric(df['t_ms'],     errors='coerce')
    df['t_std_ms'] = pd.to_numeric(df['t_std_ms'], errors='coerce')
    return df


def load_all():
    data = {}
    for key, fname in FILE_MAP.items():
        fpath = os.path.join(DATADIR, fname)
        if os.path.exists(fpath):
            data[key] = _load_csv(fpath)
        else:
            print(f'  [warn] Missing: {fname}')
    return data


def get_dim(df, dim_label):
    return df[df['dim'] == dim_label].copy().sort_values('n_pts')


# ── Drawing helpers ───────────────────────────────────────────────────────────


def _plot_backend(ax, df, bk, dim):
    """Plot one backend's time-vs-N curve."""
    sub = get_dim(df, dim)
    if sub.empty or sub['t_ms'].isna().all():
        return
    s = STYLE[bk]
    x = sub['n_pts'].values.astype(float)
    y = sub['t_ms'].values.astype(float)
    mask = np.isfinite(y)
    x, y = x[mask], y[mask]

    ax.plot(x, y, color=s['color'], ls=s['ls'], marker=s['marker'],
            ms=s['ms'], lw=1.6, zorder=s['zorder'], label=s['label'])


def _format_loglog_axis(ax, xlabel=True, ylabel=True):
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Show tick values on every panel (not just leftmost)
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    if xlabel:
        ax.set_xlabel('Grid points $N$')
    if ylabel:
        ax.set_ylabel('Time per step (ms)')


def _backend_legend_handles(backend_list):
    return [Line2D([0], [0],
                   color=STYLE[bk]['color'],
                   ls=STYLE[bk]['ls'],
                   marker=STYLE[bk]['marker'],
                   ms=STYLE[bk]['ms'],
                   lw=1.6,
                   label=STYLE[bk]['label'])
            for bk in backend_list]


# ── GPU dispatch-floor annotation ────────────────────────────────────────────

GPU_FLOOR_MS = 10.0   # approximate minimum dispatch latency on RTX 3080 Ti


def _annotate_gpu_floor(ax, floor_ms=GPU_FLOOR_MS):
    """Draw a subtle horizontal annotation for the GPU dispatch floor."""
    xl = ax.get_xlim()
    ax.axhline(floor_ms, ls='--', lw=0.75, color='0.55', alpha=0.55, zorder=0)
    ax.text(xl[1] * 10**(-0.04), floor_ms,
            'GPU floor', ha='right', va='bottom',
            fontsize=6.5, color='0.45', alpha=0.8,
            transform=ax.transData)


# ── Figure 1 & 2: Scaling plots ───────────────────────────────────────────────

DIMS = ['1D', '2D', '3D']
ALL_BKS = ['cpu_m4', 'cpu_i9', 'cuda_f32', 'cuda_f64', 'metal']


def fig_scaling(system_key, data, outfile):
    """1×3 log-log time-vs-N figure.  A GPU dispatch-floor line is shown in the polymer figure."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=False)
    sys_label = ('Diblock Copolymer (with chain propagators)'
                 if system_key == 'polymer'
                 else 'Monomeric Solvent (no propagators)')

    for col, dim in enumerate(DIMS):
        ax = axes[col]
        for bk in ALL_BKS:
            key = (bk, system_key)
            if key in data:
                _plot_backend(ax, data[key], bk, dim)

        _format_loglog_axis(ax, xlabel=True, ylabel=True)
        ax.set_title(dim, fontweight='bold', pad=4)
        ax.grid(True, which='minor', alpha=0.12, lw=0.4)

        # Add GPU floor only to polymer 2D and 3D (where it's visible and relevant)
        if system_key == 'polymer' and dim in ('2D', '3D'):
            _annotate_gpu_floor(ax)

    handles = _backend_legend_handles(ALL_BKS)
    fig.legend(handles=handles, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.18), framealpha=0.92)
    fig.suptitle(f'SCFT Step Time — {sys_label}',
                 y=1.02, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 3: Speedup over CPU i9 ────────────────────────────────────────────

GPU_BKS = ['cpu_m4', 'cuda_f32', 'cuda_f64', 'metal']


def fig_speedup(data, outfile):
    """2×3 speedup (t_i9 / t_backend) — row 0 = polymer, row 1 = solvent."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0),
                             gridspec_kw=dict(hspace=0.45, wspace=0.32))
    sys_keys   = ['polymer', 'solvent']
    sys_labels = ['Polymer (diblock copolymer)', 'Solvent (monomeric)']

    for row, system_key in enumerate(sys_keys):
        for col, dim in enumerate(DIMS):
            ax = axes[row, col]

            ref_key = ('cpu_i9', system_key)
            if ref_key not in data:
                ax.text(0.5, 0.5, 'No i9 data', ha='center', va='center',
                        transform=ax.transAxes, color='0.5')
                continue

            ref_sub = get_dim(data[ref_key], dim).dropna(subset=['t_ms'])
            if ref_sub.empty:
                continue
            ref_interp = dict(zip(ref_sub['n_pts'], ref_sub['t_ms']))

            for bk in GPU_BKS:
                key = (bk, system_key)
                if key not in data:
                    continue
                sub = get_dim(data[key], dim).dropna(subset=['t_ms'])
                if sub.empty:
                    continue
                common = sorted(set(sub['n_pts']) & set(ref_interp))
                if not common:
                    continue
                x      = np.array(common, dtype=float)
                y_ref  = np.array([ref_interp[p] for p in common])
                y_bk   = sub.set_index('n_pts').loc[common, 't_ms'].values
                speedup = y_ref / y_bk

                s = STYLE[bk]
                ax.plot(x, speedup, color=s['color'], ls=s['ls'],
                        marker=s['marker'], ms=s['ms'], lw=1.6,
                        zorder=s['zorder'], label=s['label'])

            # Parity and 10× reference lines
            ax.axhline(1.0,  ls='-',  lw=0.9, color='0.35', alpha=0.65, zorder=0)
            ax.axhline(10.0, ls=':',  lw=0.7, color='0.55', alpha=0.45, zorder=0)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
            ax.grid(True, which='minor', alpha=0.12, lw=0.4)

            # Axis labels
            if col == 0:
                ax.set_ylabel('Speedup over i9-10850K')
            if row == 1:
                ax.set_xlabel('Grid points $N$')
            if row == 0:
                ax.set_title(dim, fontweight='bold', pad=4)

            # Annotate parity line inside the plot at the left edge
            xl = ax.get_xlim()
            for level, label in [(1.0, 'parity (1×)'), (10.0, '10×')]:
                yl = ax.get_ylim()
                if yl[0] < level < yl[1]:
                    ax.text(xl[0] * 10**0.06, level * 1.12,
                            label, va='bottom', ha='left',
                            fontsize=6.5, color='0.40', alpha=0.85)

        # Row label on far right
        axes[row, 2].annotate(
            sys_labels[row],
            xy=(1.05, 0.5), xycoords='axes fraction',
            fontsize=8, rotation=-90, va='center', ha='left',
        )

    handles = _backend_legend_handles(GPU_BKS)
    fig.legend(handles=handles, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, -0.05), framealpha=0.92)
    fig.suptitle('Speedup Relative to CPU (i9-10850K, Float64)',
                 y=1.01, fontsize=10, fontweight='bold')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 4: CUDA Float32 vs Float64 ────────────────────────────────────────

def fig_cuda_precision(data, outfile):
    """2×3 — top = polymer, bottom = solvent; F32 vs F64 on CUDA."""
    try:
        prec_c = palettable.colorbrewer.qualitative.Set1_3.mpl_colors[:2]
    except Exception:
        prec_c = [_tab[2], _tab[3]]

    PREC = {
        'cuda_f32': dict(label='Float32', color=prec_c[0], ls='-',  marker='^', ms=5),
        'cuda_f64': dict(label='Float64', color=prec_c[1], ls='--', marker='v', ms=5),
    }

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey='row',
                             gridspec_kw=dict(hspace=0.42, wspace=0.25))
    sys_labels = ['Polymer (diblock copolymer)', 'Solvent (monomeric)']

    for row, system_key in enumerate(['polymer', 'solvent']):
        for col, dim in enumerate(DIMS):
            ax = axes[row, col]

            for bk, ps in PREC.items():
                key = (bk, system_key)
                if key not in data:
                    continue
                sub = get_dim(data[key], dim).dropna(subset=['t_ms'])
                if sub.empty:
                    continue
                x = sub['n_pts'].values.astype(float)
                y = sub['t_ms'].values.astype(float)
                ax.plot(x, y, color=ps['color'], ls=ps['ls'],
                        marker=ps['marker'], ms=ps['ms'], lw=1.6,
                        label=ps['label'])

            _format_loglog_axis(ax, xlabel=(row == 1), ylabel=(col == 0))
            ax.grid(True, which='minor', alpha=0.12, lw=0.4)
            if row == 0:
                ax.set_title(dim, fontweight='bold', pad=4)

            # F64/F32 ratio annotation
            f32k = ('cuda_f32', system_key)
            f64k = ('cuda_f64', system_key)
            if f32k in data and f64k in data:
                s32 = get_dim(data[f32k], dim).dropna(subset=['t_ms'])
                s64 = get_dim(data[f64k], dim).dropna(subset=['t_ms'])
                common = sorted(set(s32['n_pts']) & set(s64['n_pts']))
                if common:
                    r = np.nanmean(s64.set_index('n_pts').loc[common, 't_ms'].values /
                                   s32.set_index('n_pts').loc[common, 't_ms'].values)
                    ax.text(0.97, 0.05, f'F64/F32 = {r:.2f}×',
                            transform=ax.transAxes, ha='right', va='bottom',
                            fontsize=7, color='0.35',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      ec='0.75', alpha=0.88))

        axes[row, 2].annotate(
            sys_labels[row],
            xy=(1.05, 0.5), xycoords='axes fraction',
            fontsize=8, rotation=-90, va='center', ha='left',
        )

    axes[0, 1].legend(loc='upper left', title='CUDA Precision', title_fontsize=8)
    fig.suptitle('CUDA Float32 vs Float64 — NVIDIA RTX 3080 Ti',
                 y=1.01, fontsize=10, fontweight='bold')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 5: Throughput ──────────────────────────────────────────────────────

def fig_throughput(system_key, data, outfile):
    """1×3 throughput in millions of grid points per SCFT step (polymer)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=False)
    sys_label = 'Diblock Copolymer' if system_key == 'polymer' else 'Monomeric Solvent'

    for col, dim in enumerate(DIMS):
        ax = axes[col]
        for bk in ALL_BKS:
            key = (bk, system_key)
            if key not in data:
                continue
            sub = get_dim(data[key], dim).dropna(subset=['t_ms'])
            if sub.empty:
                continue
            x = sub['n_pts'].values.astype(float)
            t = sub['t_ms'].values.astype(float)
            tp = x / t * 1e-3                              # Mpts/s

            s = STYLE[bk]
            ax.plot(x, tp, color=s['color'], ls=s['ls'], marker=s['marker'],
                    ms=s['ms'], lw=1.6, zorder=s['zorder'], label=s['label'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.set_title(dim, fontweight='bold', pad=4)
        ax.set_xlabel('Grid points $N$')
        if col == 0:
            ax.set_ylabel('Throughput (Mpts / step)')
        ax.grid(True, which='minor', alpha=0.12, lw=0.4)

    fig.legend(handles=_backend_legend_handles(ALL_BKS),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.18),
               framealpha=0.92)
    fig.suptitle(f'Throughput — {sys_label}',
                 y=1.02, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 6: Summary bar chart ───────────────────────────────────────────────

CASES = [('1D', 1024, '1D,  $10^3$'), ('2D', 256, r'2D,  $256^2$'), ('3D', 64, r'3D,  $64^3$')]
BK_LIST = ['cpu_m4', 'cpu_i9', 'cuda_f32', 'cuda_f64', 'metal']


def fig_summary_bars(data, outfile):
    """Grouped bar chart at three representative (dim, grid-size) cases."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=False)
    bw = 0.14
    x_pos = np.arange(len(CASES))
    offs = np.linspace(-(len(BK_LIST)-1)/2, (len(BK_LIST)-1)/2, len(BK_LIST)) * bw

    for col, system_key in enumerate(['polymer', 'solvent']):
        ax = axes[col]
        for i, bk in enumerate(BK_LIST):
            vals, errs = [], []
            for dim, ng, _ in CASES:
                key = (bk, system_key)
                if key not in data:
                    vals.append(np.nan); errs.append(0); continue
                row = data[key][(data[key]['dim'] == dim) & (data[key]['ng'] == ng)]
                if row.empty:
                    vals.append(np.nan); errs.append(0)
                else:
                    vals.append(row['t_ms'].values[0])
                    e = row['t_std_ms'].values[0]
                    errs.append(e if np.isfinite(e) else 0.0)

            s = STYLE[bk]
            ax.bar(x_pos + offs[i], vals, bw, yerr=errs, capsize=2.5,
                   color=s['color'], alpha=0.85, label=s['label'],
                   error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor='0.3'))

        ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[2] for c in CASES])
        ax.set_ylabel('Time per step (ms)')
        ax.set_title('Diblock Copolymer' if system_key == 'polymer' else 'Monomeric Solvent',
                     fontweight='bold', pad=4)
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.grid(True, axis='y', which='both', alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    fig.legend(handles=[Patch(color=STYLE[bk]['color'], alpha=0.85,
                               label=STYLE[bk]['label']) for bk in BK_LIST],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.14),
               framealpha=0.92)
    fig.suptitle('Backend Comparison at Representative Grid Sizes',
                 y=1.01, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 7: Propagator overhead (polymer / solvent ratio) ──────────────────

def fig_propagator_overhead(data, outfile):
    """1×3 ratio polymer time / solvent time — quantifies chain-propagation cost."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))

    for col, dim in enumerate(DIMS):
        ax = axes[col]
        for bk in ALL_BKS:
            pk = (bk, 'polymer')
            sk = (bk, 'solvent')
            if pk not in data or sk not in data:
                continue
            sp = get_dim(data[pk], dim).dropna(subset=['t_ms'])
            ss = get_dim(data[sk], dim).dropna(subset=['t_ms'])
            common = sorted(set(sp['n_pts']) & set(ss['n_pts']))
            if not common:
                continue
            tp = sp.set_index('n_pts').loc[common, 't_ms'].values
            ts = ss.set_index('n_pts').loc[common, 't_ms'].values
            ratio = tp / np.maximum(ts, 1e-12)
            x = np.array(common, dtype=float)

            s = STYLE[bk]
            ax.plot(x, ratio, color=s['color'], ls=s['ls'],
                    marker=s['marker'], ms=s['ms'], lw=1.6,
                    zorder=s['zorder'], label=s['label'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.axhline(1.0, ls='--', lw=0.8, color='0.45', alpha=0.6, zorder=0)
        ax.set_title(dim, fontweight='bold', pad=4)
        ax.set_xlabel('Grid points $N$')
        if col == 0:
            ax.set_ylabel('Time ratio (polymer / solvent)')
        ax.grid(True, which='minor', alpha=0.12, lw=0.4)
        # Annotate the parity line
        xl = ax.get_xlim()
        ax.text(xl[0] * 10**0.06, 1.0 * 1.12, '1× (parity)',
                va='bottom', ha='left', fontsize=6.5, color='0.42', alpha=0.85)

    fig.legend(handles=_backend_legend_handles(ALL_BKS),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.18),
               framealpha=0.92)
    fig.suptitle('Chain Propagation Overhead: Polymer vs Solvent Step Time',
                 y=1.02, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Figure 8: Combined overview (2×3) ────────────────────────────────────────

def fig_combined_overview(data, outfile):
    """Compact 2×3 — top row = polymer, bottom row = solvent."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2),
                             gridspec_kw=dict(hspace=0.42, wspace=0.32))
    sys_keys   = ['polymer', 'solvent']
    sys_labels = ['Polymer (diblock copolymer)', 'Solvent (monomeric)']

    for row, system_key in enumerate(sys_keys):
        for col, dim in enumerate(DIMS):
            ax = axes[row, col]
            for bk in ALL_BKS:
                key = (bk, system_key)
                if key in data:
                    _plot_backend(ax, data[key], bk, dim)
            _format_loglog_axis(ax, xlabel=(row == 1), ylabel=True)
            ax.grid(True, which='minor', alpha=0.10, lw=0.35)
            if row == 0:
                ax.set_title(dim, fontweight='bold', pad=4)

        axes[row, 2].annotate(
            sys_labels[row],
            xy=(1.05, 0.5), xycoords='axes fraction',
            fontsize=8, rotation=-90, va='center', ha='left',
        )

    fig.legend(handles=_backend_legend_handles(ALL_BKS),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.92)
    fig.suptitle('SCFT Iteration Timing — All Backends & System Types',
                 y=1.02, fontsize=10, fontweight='bold')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))

    print('Loading benchmark CSV files...')
    data = load_all()
    if not data:
        sys.exit('No data files found. Run benchmarks first.')
    print(f'  Loaded {len(data)} datasets.\n')

    def out(name):
        return os.path.join(outdir, name)

    print('Generating figures...')
    fig_scaling('polymer', data, out('fig1_polymer_scaling.pdf'))
    fig_scaling('solvent', data, out('fig2_solvent_scaling.pdf'))
    fig_speedup(data,            out('fig3_speedup.pdf'))
    fig_cuda_precision(data,     out('fig4_cuda_precision.pdf'))
    fig_throughput('polymer', data, out('fig5_throughput.pdf'))
    fig_summary_bars(data,       out('fig6_summary_bars.pdf'))
    fig_propagator_overhead(data, out('fig7_propagator_overhead.pdf'))
    fig_combined_overview(data,  out('fig8_combined_overview.pdf'))

    print(f'\nAll figures written to {outdir}/')


if __name__ == '__main__':
    main()
