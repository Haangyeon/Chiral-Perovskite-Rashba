#!/usr/bin/env python3
"""Rebuild the two legacy diagnostic figures with submission-safe layouts.

This script reads only the archived post-processed CSV files.  It does not run
VASP or alter any scientific result.  The visual proportions follow the PCCP
submission figures, while labels are cleaned up for the revised manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "lightgray": "#D9D9D9",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.6,
            "axes.labelsize": 9.0,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def save_all(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    """Save submission PDF/PNG plus an editable SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_layer_share(input_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_dir / "layer_spin_D3.csv")
    order = ["S", "R", "Rac", "3ClMBA", "4BrMBA"]
    data["system"] = pd.Categorical(data["system"], categories=order, ordered=True)
    data = data.sort_values("system").copy()
    data["Pb-I framework"] = 100.0 * data["ratio_inorganic_over_total"]
    data["organic spacer"] = 100.0 - data["Pb-I framework"]

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.15, 3.6))
    ax.barh(
        y,
        data["Pb-I framework"],
        height=0.72,
        color=COLORS["blue"],
        edgecolor="white",
        label="Pb-I framework",
    )
    ax.barh(
        y,
        data["organic spacer"],
        left=data["Pb-I framework"],
        height=0.72,
        color=COLORS["lightgray"],
        edgecolor="white",
        label="Organic spacer",
    )
    for yi, share in zip(y, data["Pb-I framework"]):
        ax.text(
            min(float(share) - 1.2, 97.0),
            yi,
            f"{share:.1f}%",
            ha="right",
            va="center",
            fontsize=8.2,
            color="white",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["system"].astype(str))
    ax.invert_yaxis()
    ax.set_ylim(-0.65, 4.65)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Projected spin share (%)")
    # Grid lines removed for the revised manuscript (requested 2026-09-15):
    # ax.xaxis.grid(True, color="#E8E8E8", lw=0.55)
    ax.set_axisbelow(True)
    left, right = 0.13, 0.985
    data_center = 0.5 * (left + right)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(data_center, 0.99),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=left, right=right, bottom=0.14, top=0.91)
    save_all(fig, output_dir, "Fig5layer_spin_D3_core_iop_v5")


def find_pairs(data: pd.DataFrame, digits: int = 8) -> list[tuple[int, int]]:
    lookup = {
        (round(float(row.kx_cart), digits), round(float(row.ky_cart), digits)): idx
        for idx, row in data.iterrows()
    }
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for idx, row in data.iterrows():
        if idx in used:
            continue
        partner = lookup.get(
            (round(-float(row.kx_cart), digits), round(-float(row.ky_cart), digits))
        )
        if partner is not None and partner != idx and partner not in used:
            pairs.append((idx, partner))
            used.add(idx)
            used.add(partner)
    return pairs


def safe_corr(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def plot_time_reversal(input_dir: Path, output_dir: Path) -> None:
    systems = ["R", "S", "Rac"]
    system_titles = {
        "R": r"($R$)-3BrMBA$_2$PbI$_4$",
        "S": r"($S$)-3BrMBA$_2$PbI$_4$",
        "Rac": r"rac-(3BrMBA)$_2$PbI$_4$",
    }
    component_colors = {
        "Sx": COLORS["orange"],
        "Sy": COLORS["blue"],
        "Sz": COLORS["green"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.60))
    legend_handles = None
    legend_labels = None
    for panel, (ax, system) in enumerate(zip(axes, systems)):
        data = pd.read_csv(input_dir / f"{system}_spin_texture.csv")
        pairs = find_pairs(data)
        if not pairs:
            raise RuntimeError(f"No +/-k pairs found for {system}")
        positive = data.loc[[pair[0] for pair in pairs]]
        negative = data.loc[[pair[1] for pair in pairs]]

        for column, color in component_colors.items():
            ax.scatter(
                positive[column],
                negative[column],
                s=12,
                alpha=0.68,
                color=color,
                edgecolors="none",
                label=rf"${column[0]}_{column[1]}$",
            )
        maximum = max(
            float(np.nanmax(np.abs(positive[["Sx", "Sy", "Sz"]].to_numpy(float)))),
            float(np.nanmax(np.abs(negative[["Sx", "Sy", "Sz"]].to_numpy(float)))),
            0.02,
        )
        limit = 1.12 * maximum
        ax.plot([-limit, limit], [limit, -limit], "--", lw=0.85, color="0.15", label=r"$y=-x$")
        ax.plot([-limit, limit], [-limit, limit], ":", lw=0.85, color="0.48", label=r"$y=x$")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$S_i(\mathbf{k})$")
        if panel == 0:
            ax.set_ylabel(r"$S_i(-\mathbf{k})$")

        corr_x = safe_corr(
            positive["Sx"].to_numpy(float),
            -negative["Sx"].to_numpy(float),
        )
        corr_z = safe_corr(
            positive["Sz"].to_numpy(float),
            -negative["Sz"].to_numpy(float),
        )
        if system == "Rac":
            subtitle = r"$PT$-pair average = 0"
        else:
            subtitle = rf"$T$-pair correlation: $S_x$ {corr_x:.2f}; $S_z$ {corr_z:.2f}"
        ax.set_title(
            f"({chr(97 + panel)})  " + system_titles[system] + "\n" + subtitle,
            pad=5,
            y=1.01,
            fontsize=8.6,
            fontweight="normal",
            linespacing=1.12,
        )
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    left, right = 0.075, 0.985
    data_center = 0.5 * (left + right)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(data_center, 0.025),
        bbox_transform=fig.transFigure,
        ncol=5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=left, right=right, bottom=0.22, top=0.77, wspace=0.32)
    save_all(fig, output_dir, "time_reversal_symmetry_v20")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    configure_style()
    plot_layer_share(args.input_dir, args.output_dir)
    plot_time_reversal(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
