#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_si_jana_descriptor_schematic.py
====================================
SI figure: definition of the asymmetric in-plane octahedral-tilting disparity
Delta_beta_in introduced by Jana et al. (Nat. Commun. 12, 4982 (2021)).

  (a) symmetric in-plane tilting  -> a single bridge angle (beta = beta'), Delta_beta_in = 0
  (b) asymmetric in-plane tilting -> two bridge angles (beta, beta'), Delta_beta_in = |beta - beta'|

The in-plane component beta_in is obtained by projecting the equatorial halogen
onto the horizontal plane containing the M (Pb) atoms.  Outputs PDF + SVG + PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Arc

HERE = Path(__file__).resolve().parent

# Outer whitespace: every side is padded to the same *fraction of the final
# page*, which is the rule the other revised figures follow and the only one
# that makes two differently-shaped figures agree at the edges.  Fig. S4 uses
# the same value.
EDGE_MARGIN = 0.020


def _ink_bbox_pt(page, dpi=600.0, thresh=250):
    """Ink bounding box of a PDF page, in points, y measured downward."""
    import pymupdf

    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].min(axis=2)
    mask = arr < thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    s = 72.0 / dpi
    return (cols.min() * s, rows.min() * s,
            (cols.max() + 1) * s, (rows.max() + 1) * s)


def save_cropped(fig, stem, edge=EDGE_MARGIN, dpi=300):
    """Write `<stem>.pdf/.png/.svg` with a uniform *relative* outer margin.

    Two tempting shortcuts do not work here and both were measured:

    * ``pad_inches`` pads by a fixed number of points, so on this wide, short
      figure the top and bottom bands come out proportionally larger than the
      left and right ones.  Padding enough to reach the 2 % left/right target
      pushes the top/bottom bands past the 3 % ceiling.
    * Passing an explicit crop box to ``savefig`` makes matplotlib re-solve the
      constrained layout for the new canvas, which shrank the panels by 1.8 %
      and widened the inter-panel gap -- a layout change, not a margin change.

    So the figure is rendered exactly as matplotlib would render it
    (``bbox_inches='tight'``) and only then re-framed at the page level, with
    the artwork pasted at scale 1.  Panels, fonts and text positions are
    untouched; only the four outer bands move.  This also fixes the previously
    clipped title, which the old fixed-height canvas cut into.
    """
    import pymupdf

    stem = Path(stem)
    pdf, png, svg = (stem.with_suffix(e) for e in ('.pdf', '.png', '.svg'))
    scratch = stem.with_suffix('.scratch.pdf')
    fig.savefig(scratch, bbox_inches='tight', pad_inches=0)

    doc = pymupdf.open(scratch)
    src = doc[0]
    ix0, iy0, ix1, iy1 = _ink_bbox_pt(src)
    new_w = (ix1 - ix0) / (1.0 - 2.0 * edge)
    new_h = (iy1 - iy0) / (1.0 - 2.0 * edge)
    ox, oy = edge * new_w - ix0, edge * new_h - iy0
    out = pymupdf.open()
    page = out.new_page(width=new_w, height=new_h)
    page.show_pdf_page(pymupdf.Rect(ox, oy, ox + src.rect.width, oy + src.rect.height),
                       doc, 0, clip=src.rect)
    out.save(pdf)
    out.close()
    doc.close()

    # Rasterize the cropped page so the PNG cannot disagree with the PDF.
    doc = pymupdf.open(pdf)
    doc[0].get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0),
                      alpha=False).save(png)
    doc.close()

    # SVG is an auxiliary editable format: re-frame it onto the same crop.
    try:
        import re
        fig.savefig(svg, bbox_inches='tight', pad_inches=0)
        text = svg.read_text(encoding='utf-8')
        text = re.sub(r'width="[\d.]+pt"', f'width="{new_w:.2f}pt"', text, count=1)
        text = re.sub(r'height="[\d.]+pt"', f'height="{new_h:.2f}pt"', text, count=1)
        text = re.sub(r'viewBox="[^"]*"',
                      f'viewBox="{-ox:.2f} {-oy:.2f} {new_w:.2f} {new_h:.2f}"',
                      text, count=1)
        svg.write_text(text, encoding='utf-8')
    except Exception:  # noqa: BLE001
        pass

    if scratch.exists():
        scratch.unlink()


def resolve_out():
    # FIGS3_OUTDIR selects a sandbox output directory; without it the original
    # preference order (the manuscript's all_figures/) is kept.
    import os
    override = os.environ.get('FIGS3_OUTDIR')
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in [HERE.parent.parent / 'all_figures', HERE.parent / 'all_figures',
                 HERE]:
        if cand.is_dir():
            return cand
    return HERE

# Match the font stack used by the main-text figures (Fig. 6 / Fig. S4) so the
# Greek symbols and Latin labels resolve to the same families across figures.
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Nimbus Sans']
rcParams['font.size'] = 8.5
rcParams['axes.labelsize'] = 8.5
rcParams['xtick.labelsize'] = 7.5
rcParams['ytick.labelsize'] = 7.5
rcParams['pdf.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['axes.unicode_minus'] = False

PB, XC = '#4d4d4d', '#7b3294'


def angle_at(vertex, p1, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def draw_panel(ax, x_positions):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.80, 1.85)
    ax.set_box_aspect(0.80)

    pb1, pb2 = (-1.15, 0.0), (1.15, 0.0)
    # M plane (horizontal plane containing the Pb atoms)
    ax.plot([-1.95, 1.95], [0, 0], color='#bbbbbb', lw=0.8, ls=(0, (5, 4)), zorder=1)
    ax.text(1.95, -0.16, 'plane of M (Pb) atoms', ha='right', va='top',
            fontsize=7.0, color='#777777')

    for m in (pb1, pb2):
        ax.scatter(*m, s=150, c=PB, zorder=4, edgecolors='white', linewidths=0.6)
    ax.text(pb1[0], pb1[1] - 0.30, 'Pb', ha='center', va='top', fontsize=7.6)
    ax.text(pb2[0], pb2[1] - 0.30, 'Pb', ha='center', va='top', fontsize=7.6)

    for k, xp in enumerate(x_positions):
        xc = (xp[0], xp[1])
        ax.plot([pb1[0], xc[0]], [pb1[1], xc[1]], color='#888888', lw=0.9, zorder=2)
        ax.plot([pb2[0], xc[0]], [pb2[1], xc[1]], color='#888888', lw=0.9, zorder=2)
        ax.scatter(*xc, s=95, c=XC, marker='o', zorder=5,
                   edgecolors='white', linewidths=0.6)
        label = 'X' if len(x_positions) == 1 else ('X' if k == 0 else "X′")
        ax.text(xc[0], xc[1] + 0.20, label, ha='center', va='bottom',
                fontsize=7.6, color=XC, fontweight='bold')
        # angle arc at X
        ang = angle_at(xc, pb1, pb2)
        a1 = np.degrees(np.arctan2(pb1[1] - xc[1], pb1[0] - xc[0]))
        a2 = np.degrees(np.arctan2(pb2[1] - xc[1], pb2[0] - xc[0]))
        lo, hi = min(a1, a2), max(a1, a2)
        ax.add_patch(Arc(xc, 0.62, 0.62, theta1=lo, theta2=hi,
                         color='#c0392b', lw=1.0, zorder=3))
        mid = np.radians(0.5 * (lo + hi))
        ax.text(xc[0] + 0.42 * np.cos(mid), xc[1] + 0.42 * np.sin(mid),
                (r'$\beta$' if len(x_positions) == 1 or k == 0 else r"$\beta'$"),
                ha='center', va='center', fontsize=9.5, color='#c0392b', zorder=6)
    return pb1, pb2


def main():
    OUT = resolve_out()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), layout='constrained')

    draw_panel(axes[0], [(0.0, 1.35)])
    # 2026-09-23: centred and non-bold, to match the panel-title convention
    # used by Fig. 6 and the other multi-panel figures.  `loc='left'` anchored
    # the title to the figure edge rather than to the panel, so (a) sat 73 pt
    # left of its panel centre and (b) 45 pt right of its own.
    axes[0].set_title('(a)  Symmetric tilting', loc='center', fontsize=8.5,
                      fontweight='normal', pad=4)
    axes[0].annotate(r'$\beta = \beta^{\prime}$;   $\Delta\beta_{\mathrm{in}} = 0$',
                     xy=(0.0, -0.30), xytext=(0.0, -0.62),
                     ha='center', va='top', fontsize=8.5, color='#c0392b')

    draw_panel(axes[1], [(-0.30, 1.42), (0.30, 1.12)])
    axes[1].set_title('(b)  Asymmetric tilting', loc='center', fontsize=8.5,
                      fontweight='normal', pad=4)
    axes[1].annotate(r'$\Delta\beta_{\mathrm{in}} = |\beta - \beta^{\prime}| > 0$',
                     xy=(0.0, -0.30), xytext=(0.0, -0.62),
                     ha='center', va='top', fontsize=8.5, color='#c0392b')

    stem = OUT / 'FigS_jana_descriptor_schematic'
    save_cropped(fig, stem)
    for ext in ('pdf', 'svg', 'png'):
        print('wrote', stem.with_suffix('.' + ext))
    plt.close(fig)


if __name__ == '__main__':
    main()
