#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_si_jana_consistency.py
===========================
SI figure: consistency checks for the Delta_beta_in -- alpha_R^ext association.

  (a) k0   vs Delta_beta_in      (component of alpha_R^ext = 2 E_R / k0)
  (b) E_R  vs Delta_beta_in      (component of alpha_R^ext = 2 E_R / k0)
  (c) projection-definition sensitivity: Delta_beta_in from the coplanar-Pb-plane
      projection vs the axial I-I projection (y = x guide line).

k0 and E_R are mathematically dependent components of alpha_R^ext and are shown
only as consistency checks, not as independent findings.  Outputs PDF + SVG + PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent


def find_data(name: str):
    for cand in [HERE, HERE / 'data', HERE / 'descriptors', HERE.parent,
                 HERE.parent / 'descriptors', HERE.parent / 'data',
                 HERE.parent.parent, HERE.parent.parent / 'descriptors',
                 Path.cwd()]:
        p = cand / name
        if p.is_file():
            return p
    raise SystemExit(f'Could not locate {name}')


def resolve_out():
    for cand in [HERE.parent.parent / 'all_figures', HERE.parent / 'all_figures',
                 HERE]:
        if cand.is_dir():
            return cand
    return HERE

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 8.5
rcParams['axes.labelsize'] = 8.5
rcParams['xtick.labelsize'] = 7.5
rcParams['ytick.labelsize'] = 7.5
rcParams['axes.linewidth'] = 0.7
rcParams['pdf.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['axes.unicode_minus'] = False

STYLE = {
    'S': ('#d62728', 'o'), 'R': ('#d62728', 's'), 'Rac': ('#1f77b4', 'D'),
    '2BrMBA': ('#ff7f0e', '*'), '3ClMBA': ('#2ca02c', '^'),
    '3FMBA': ('#2ca02c', 'v'), '3IMBA': ('#2ca02c', '<'),
    '4BrMBA': ('#2ca02c', '>'), 'MBA': ('#9467bd', 'p'),
}


def fmt_p(p):
    m, e = f'{p:.1e}'.split('e')
    return rf'${m}\times10^{{{int(e)}}}$'


def style_ax(ax):
    ax.tick_params(direction='out', length=3.0, width=0.7, pad=2.5)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_box_aspect(0.80)          # panel w:h = 1.25


def panel(ax, d, xcol, ycol, ylab, title, stats, diag=False):
    style_ax(ax)
    for _, r in d.iterrows():
        c, m = STYLE[r['system']]
        size = 46 if r['system'] == '2BrMBA' else 30
        ax.scatter(r[xcol], r[ycol], s=size, c=c, marker=m,
                   edgecolors='white', linewidths=0.55, zorder=4)
    texts = [ax.text(r[xcol], r[ycol], r['system'], fontsize=6.3, zorder=6,
                     ha='center', va='center', color='#111111')
             for _, r in d.iterrows()]

    xv, yv = d[xcol].to_numpy(float), d[ycol].to_numpy(float)
    xpad = 0.06 * (xv.max() - xv.min())
    ypad = 0.10 * (yv.max() - yv.min())
    ax.set_xlim(xv.min() - xpad, xv.max() + xpad)
    ax.set_ylim(yv.min() - ypad, yv.max() + ypad)

    if diag:
        lo = min(xv.min(), yv.min())
        hi = max(xv.max(), yv.max())
        ax.plot([lo, hi], [lo, hi], color='#888888', lw=0.7, ls='--', zorder=2)

    ax.set_xlabel(r'$\Delta\beta_{\mathrm{in}}$ ($^\circ$)')
    ax.set_ylabel(ylab)
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', pad=5)
    stat_txt = ax.text(0.975, 0.035, stats, transform=ax.transAxes, fontsize=7.0,
                       ha='right', va='bottom', zorder=8, linespacing=1.35)
    adjust_text(texts, ax=ax, objects=[stat_txt],
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5),
                expand=(1.3, 1.5), force_text=(0.4, 0.6), force_static=(0.4, 0.6),
                force_explode=(0.2, 0.7), force_pull=(0.02, 0.02),
                ensure_inside_axes=True, iter_lim=500, min_arrow_len=3)


def main():
    OUT = resolve_out()
    d = pd.read_csv(find_data('jana_rashba_validation_merged_final.csv'))
    from scipy import stats
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.95), layout='constrained')

    r, p = stats.pearsonr(d.delta_beta_in_plane, d.k0)
    rho, prho = stats.spearmanr(d.delta_beta_in_plane, d.k0)
    panel(axes[0], d, 'delta_beta_in_plane', 'k0',
          r'$k_0$ ($\mathrm{\AA}^{-1}$)', '(a)  $k_0$ (component)',
          f"$r$ = {r:.3f}, $p$ = {fmt_p(p)}\n$\\rho$ = {rho:.3f}")

    r, p = stats.pearsonr(d.delta_beta_in_plane, d.E_R)
    rho, prho = stats.spearmanr(d.delta_beta_in_plane, d.E_R)
    panel(axes[1], d, 'delta_beta_in_plane', 'E_R',
          r'$E_R$ (eV)', '(b)  $E_R$ (component)',
          f"$r$ = {r:.3f}, $p$ = {fmt_p(p)}\n$\\rho$ = {rho:.3f}")

    r, p = stats.pearsonr(d.delta_beta_in_plane, d.delta_beta_in_axial)
    panel(axes[2], d, 'delta_beta_in_plane', 'delta_beta_in_axial',
          r'$\Delta\beta_{\mathrm{in}}^{\mathrm{axial}}$ ($^\circ$)',
          '(c)  Projection sensitivity',
          f"$r$ = {r:.4f}, $p$ = {fmt_p(p)}", diag=True)

    for ext in ('pdf', 'svg', 'png'):
        pp = OUT / f'FigS_jana_consistency_and_sensitivity.{ext}'
        fig.savefig(pp, dpi=(300 if ext == 'png' else None))
        print('wrote', pp)
    plt.close(fig)


if __name__ == '__main__':
    main()
