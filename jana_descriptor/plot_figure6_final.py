#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figure6_final.py
=====================
Main-text Figure 6 (2 x 2):
  (a) in-plane Pb off-centering  delta_par   vs |S|_tot
  (b) mean Pb-I-Pb bridge bending            vs |S|_tot
  (c) bond-length distortion index (BLDI)    vs |S|_tot
  (d) asymmetric in-plane octahedral-tilting disparity
      Delta_beta_in vs the branch-extremum Rashba coefficient
      alpha_R^ext = 2 E_R / k0

Layout rules:
  * constrained layout only (no manual wspace/hspace)
  * single panel w:h ~ 1.25 ; whole figure w:h ~ 1.15
  * scatter labels auto-decluttered with adjustText (leader lines)
  * no gridlines, thin spines, Arial, axis labels 8.5 pt, ticks 7.5 pt
  * p-values in scientific notation (never "0.0000")

LABELLING RULE (canonical, all four panels identical):
  All nine systems are labelled in every panel (a), (b), (c) and (d).
  No per-panel exception list. Overlap in the clustered region is resolved
  by adjustText leader lines, NOT by dropping labels.

  Rationale: the reader must be able to identify every data point in every
  panel on its own; a panel that labels only "outliers" forces the reader
  to cross-compare panels to learn which marker is which.  This rule is
  asserted by _dodtest/f6_criterion_A.py (visible-label identity set).

CONFIG CONTRACT (DoD v3):
  LABEL_FONT_SIZE / LABEL_FONT_FAMILY below are the single source of truth
  for the annotation tier.  QA scripts (_dodtest/f6_criterion_A.py,
  _dodtest/f6_manuscript_embed_check.py) import THIS module and read these
  constants instead of transcribing literals, so that a figure change can
  never silently drift away from the acceptance threshold.

Outputs PDF + SVG + PNG(300 dpi) into OUT_DIR (default: sibling all_figures/).
"""

from __future__ import annotations

import os
import sys
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

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for the annotation tier  (read by the QA scripts)
# ---------------------------------------------------------------------------
LABEL_FONT_SIZE = 6.8          # pt, data-point system labels, ALL four panels
LABEL_FONT_SIZE_TOL = 0.35     # pt, acceptance window used by criterion A
LABEL_FONT_FAMILY = 'Arial'    # exported as ArialMT in PDF (fonttype 42)

FIGURE_STEM = 'Fig6_descriptor_correlation_2x2'
DESCRIPTOR_COLS = ('delta_inplane', 'bridge_bending', 'BLDI')
RESPONSE_COL = 'alpha_r'
NINE_SYSTEMS = ('Rac', '2BrMBA', '3ClMBA', '3FMBA', '3IMBA',
                '4BrMBA', 'MBA', 'S', 'R')


def find_data(name: str):
    """Locate an input CSV.

    Searches, in order: the script directory, its data/descriptors/jana_descriptor
    subdirs, the parent tree (incl. jana_descriptor/ and descriptors/), and the CWD.
    ``jana_descriptor/`` is included because the merged Rashba table lives there
    in the 07 package layout (this was a latent gap in the earlier revision).
    """
    for cand in [HERE, HERE / 'data', HERE / 'descriptors',
                 HERE / 'jana_descriptor',
                 HERE.parent, HERE.parent / 'descriptors',
                 HERE.parent / 'data', HERE.parent / 'jana_descriptor',
                 HERE.parent.parent, HERE.parent.parent / 'descriptors',
                 HERE.parent.parent / 'jana_descriptor',
                 Path.cwd()]:
        p = cand / name
        if p.is_file():
            return p
    raise SystemExit(f'Could not locate {name}')


def resolve_out():
    """Output directory, in priority order.

    ``FIG6_OUTDIR`` env var  >  argv[--outdir]  >  sibling all_figures/  >  HERE
    """
    env = os.environ.get('FIG6_OUTDIR')
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    argv = sys.argv[1:]
    if '--outdir' in argv:
        p = Path(argv[argv.index('--outdir') + 1])
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in [HERE.parent.parent / 'all_figures', HERE.parent / 'all_figures',
                 HERE]:
        if cand.is_dir():
            return cand
    return HERE

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [LABEL_FONT_FAMILY, 'DejaVu Sans']
rcParams['font.size'] = 8.5
rcParams['axes.labelsize'] = 8.5
rcParams['xtick.labelsize'] = 7.5
rcParams['ytick.labelsize'] = 7.5
rcParams['axes.linewidth'] = 0.7
rcParams['pdf.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['axes.unicode_minus'] = False

STYLE = {
    'S':      ('#d62728', 'o'),
    'R':      ('#d62728', 's'),
    'Rac':    ('#1f77b4', 'D'),
    '2BrMBA': ('#ff7f0e', '*'),
    '3ClMBA': ('#2ca02c', '^'),
    '3FMBA':  ('#2ca02c', 'v'),
    '3IMBA':  ('#2ca02c', '<'),
    '4BrMBA': ('#2ca02c', '>'),
    'MBA':    ('#9467bd', 'p'),
}


def fmt_p(p):
    """Scientific notation, never 0.0000."""
    m, e = f'{p:.1e}'.split('e')
    return rf'${m}\times10^{{{int(e)}}}$'


def style_ax(ax):
    ax.tick_params(direction='out', length=3.0, width=0.7, pad=2.5)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_box_aspect(0.80)          # panel w:h = 1.25


def panel(ax, rows, xcol, ycol, xlabel, ylabel, title, stats,
          xlim, ylim, show_trend=False):
    style_ax(ax)
    for _, r in rows.iterrows():
        c, m = STYLE[r['system']]
        size = 42 if r['system'] == '2BrMBA' else 30
        ax.scatter(r[xcol], r[ycol], s=size, c=c, marker=m,
                   edgecolors='white', linewidths=0.55, zorder=4)
    # 标注策略：四格一律标全部 9 个体系，密集区由 adjustText leader line 避让
    texts = [ax.text(r[xcol], r[ycol], r['system'], fontsize=LABEL_FONT_SIZE,
                     zorder=6, ha='center', va='center', color='#111111')
             for _, r in rows.iterrows()]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', pad=5)

    if show_trend:
        x = rows[xcol].to_numpy(float)
        y = rows[ycol].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, b * xs + a, color='#888888', lw=0.7, ls='--', zorder=2)

    stat_txt = ax.text(0.975, 0.035, stats, transform=ax.transAxes,
                       fontsize=7.0, ha='right', va='bottom', zorder=8,
                       linespacing=1.35)

    adjust_text(texts, ax=ax, objects=[stat_txt],
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5),
                expand=(1.35, 1.55),
                force_text=(0.45, 0.65), force_static=(0.45, 0.65),
                force_explode=(0.25, 0.75), force_pull=(0.02, 0.02),
                only_move={'text': 'xy', 'static': 'xy',
                           'explode': 'xy', 'pull': 'xy'},
                ensure_inside_axes=True, iter_lim=500, min_arrow_len=4)
    return texts


def main():
    OUT = resolve_out()
    df6 = pd.read_csv(find_data('Figure6_authoritative_data.csv'))
    merged = pd.read_csv(find_data('jana_rashba_validation_merged_final.csv'))

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 6.10), layout='constrained')

    # ---- (a) Pb off-centering -------------------------------------------
    panel(axes[0, 0], df6, 'delta_inplane', 'mean_s_tot',
          r'$\delta_{\parallel}$ ($\mathrm{\AA}$)', r'$|S|_{\mathrm{tot}}$',
          '(a)  Pb off-centering',
          "$r$ = -0.018, $p$ = 0.963\n$\\rho$ = -0.017, $p$ = 0.966",
          (-0.012, 0.375), (0.150, 0.342))

    # ---- (b) mean bridge bending ----------------------------------------
    panel(axes[0, 1], df6, 'bridge_bending', 'mean_s_tot',
          r'$\bar{\delta}_{\mathrm{bridge}}$ ($^\circ$)', r'$|S|_{\mathrm{tot}}$',
          '(b)  Mean Pb-I-Pb bending',
          "$r$ = 0.756, $p$ = 0.018\n$\\rho$ = -0.067, $p$ = 0.860",
          (30.45, 34.62), (0.150, 0.342))

    # ---- (c) BLDI --------------------------------------------------------
    panel(axes[1, 0], df6, 'BLDI', 'mean_s_tot',
          'BLDI', r'$|S|_{\mathrm{tot}}$',
          '(c)  Bond-length distortion',
          "$r$ = -0.041, $p$ = 0.920\n$\\rho$ = 0.267, $p$ = 0.490",
          (0.0015, 0.0885), (0.150, 0.342))

    # ---- (d) Delta_beta_in vs alpha_R^ext --------------------------------
    panel(axes[1, 1], merged, 'delta_beta_in_plane', 'alpha_r',
          r'$\Delta\beta_{\mathrm{in}}$ ($^\circ$)',
          r'$\alpha_{R}^{\mathrm{ext}}$ (eV$\cdot\mathrm{\AA}$)',
          '(d)  Asymmetric-tilting disparity',
          f"$r$ = 0.990, $p$ = {fmt_p(3.7045e-7)}\n"
          f"$\\rho$ = 0.983, $p$ = {fmt_p(1.9362e-6)}",
          (-0.60, 13.65), (-0.065, 1.285), show_trend=True)

    for ext in ('pdf', 'svg', 'png'):
        p = OUT / f'{FIGURE_STEM}.{ext}'
        fig.savefig(p, dpi=(300 if ext == 'png' else None))
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    main()
