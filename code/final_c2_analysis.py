#!/usr/bin/env python3
"""Final, symmetry-constrained low-k analysis for S/R*/Rac 3BrMBA2PbI4.

The script deliberately separates three quantities:

1. the C2-constrained two-dimensional energy-splitting norm,
2. the independent Gamma-X (unit-a*) axial check, and
3. the low-k branch-resolved spin texture.

No k0 or ER values are produced because the inferred extrema lie outside the
available line-scan interval.  Centrosymmetric Rac results are reported as
unresolved (N/A) when the doublet splitting is at the eigenvalue-output floor.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


SYSTEMS = ("S", "Rstar", "Rac")
EDGES = {"valence": (303, 304), "conduction": (305, 306)}
WINDOWS = (0.005, 0.0075, 0.010)
ENERGY_QUANTUM_EV = 1.0e-6
RAC_RESOLUTION_MEV = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_map(path: Path) -> list[dict]:
    rows = read_tsv(path)
    for row in rows:
        for key in ("global_k", "ia", "ib"):
            row[key] = int(row[key])
        for key in (
            "q_a_Ainv",
            "q_b_Ainv",
            "kx_Ainv",
            "ky_Ainv",
            "kz_Ainv",
            "radius_Ainv",
        ):
            row[key] = float(row[key])
        row["radial_keep"] = bool(int(row["radial_keep"]))
    return rows


def load_bands(path: Path) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            k = int(row["global_k"])
            band = int(row["band"])
            for field in (
                "energy_eV",
                "occupation",
                "charge",
                "sx",
                "sy",
                "sz",
                "orb_s",
                "orb_p",
                "orb_d",
            ):
                row[field] = float(row[field])
            row["band"] = band
            row["element_charge"] = json.loads(row["elem_charge_json"])
            out.setdefault(k, {})[band] = row
    return out


def feature(row: dict) -> np.ndarray:
    elems = row["element_charge"]
    vector = np.array(
        [row["orb_s"], row["orb_p"], row["orb_d"]]
        + [elems.get(x, 0.0) for x in ("H", "Pb", "C", "I", "Br", "N")],
        dtype=float,
    )
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def select_pairs(mapping: list[dict], bands: dict[int, dict[int, dict]], edge: str) -> dict[int, dict]:
    ref_bands = EDGES[edge]
    gamma = min(mapping, key=lambda x: x["radius_Ainv"])["global_k"]
    ref_rows = [bands[gamma][b] for b in ref_bands]
    ref_energy = float(np.mean([row["energy_eV"] for row in ref_rows]))
    ref_feature = feature(ref_rows[0]) + feature(ref_rows[1])
    ref_feature /= np.linalg.norm(ref_feature)
    want_occupied = edge == "valence"
    selected: dict[int, dict] = {}

    for point in mapping:
        k = point["global_k"]
        candidates = []
        keys = sorted(bands[k])
        for b1, b2 in zip(keys, keys[1:]):
            row1, row2 = bands[k][b1], bands[k][b2]
            occupied = 0.5 * (row1["occupation"] + row2["occupation"]) > 0.5
            if occupied != want_occupied:
                continue
            mean_energy = 0.5 * (row1["energy_eV"] + row2["energy_eV"])
            pair_feature = feature(row1) + feature(row2)
            if np.linalg.norm(pair_feature):
                pair_feature /= np.linalg.norm(pair_feature)
            score = abs(mean_energy - ref_energy) / 0.15
            score += 3.0 * (1.0 - float(np.clip(np.dot(pair_feature, ref_feature), -1.0, 1.0)))
            candidates.append((score, row1, row2))
        if not candidates:
            raise RuntimeError(f"No {edge} pair candidate at global_k={k}")
        _, row1, row2 = min(candidates, key=lambda item: item[0])
        lower, upper = sorted((row1, row2), key=lambda row: row["energy_eV"])
        selected[k] = {
            "lower_band": lower["band"],
            "upper_band": upper["band"],
            "lower_energy": lower["energy_eV"],
            "upper_energy": upper["energy_eV"],
            "lower_spin": np.array([lower["sx"], lower["sy"], lower["sz"]]),
            "upper_spin": np.array([upper["sx"], upper["sy"], upper["sz"]]),
        }
    return selected


def arrays(mapping: list[dict], selected: dict[int, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.array([[p["q_a_Ainv"], p["q_b_Ainv"]] for p in mapping], dtype=float)
    mean_energy = np.array(
        [
            0.5
            * (
                selected[p["global_k"]]["lower_energy"]
                + selected[p["global_k"]]["upper_energy"]
            )
            for p in mapping
        ],
        dtype=float,
    )
    half_split = np.array(
        [
            0.5
            * (
                selected[p["global_k"]]["upper_energy"]
                - selected[p["global_k"]]["lower_energy"]
            )
            for p in mapping
        ],
        dtype=float,
    )
    return q, mean_energy, half_split


def c2_fit(q: np.ndarray, mean_energy: np.ndarray, half_split: np.ndarray) -> dict:
    nonzero = np.linalg.norm(q, axis=1) > 1.0e-12
    qn = q[nonzero]
    yn = half_split[nonzero]

    def residual(params: np.ndarray) -> np.ndarray:
        alpha_a, alpha_b = params
        prediction = np.sqrt((alpha_a * qn[:, 0]) ** 2 + (alpha_b * qn[:, 1]) ** 2)
        return prediction - yn

    axis_a = np.abs(q[:, 1]) < 1.0e-12
    axis_b = np.abs(q[:, 0]) < 1.0e-12
    x_a = np.abs(q[axis_a, 0])
    x_b = np.abs(q[axis_b, 1])
    y_a = half_split[axis_a]
    y_b = half_split[axis_b]
    nz_a = x_a > 1.0e-12
    nz_b = x_b > 1.0e-12
    initial = np.array(
        [
            np.dot(x_a[nz_a], y_a[nz_a]) / np.dot(x_a[nz_a], x_a[nz_a]),
            np.dot(x_b[nz_b], y_b[nz_b]) / np.dot(x_b[nz_b], x_b[nz_b]),
        ]
    )
    fit = least_squares(residual, initial, bounds=(0.0, np.inf), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    alpha_a, alpha_b = (float(x) for x in fit.x)
    prediction = np.sqrt((alpha_a * q[:, 0]) ** 2 + (alpha_b * q[:, 1]) ** 2)

    design = np.column_stack((np.ones(len(q)), q[:, 0] ** 2, q[:, 1] ** 2))
    mean_coeff = np.linalg.lstsq(design, mean_energy, rcond=None)[0]
    mean_prediction = design @ mean_coeff

    return {
        "alpha_a_star_eVA": alpha_a,
        "alpha_b_star_eVA": alpha_b,
        "anisotropy_alpha_a_over_b": alpha_a / alpha_b if alpha_b > 0 else None,
        "split_half_rmse_meV": float(1000.0 * np.sqrt(np.mean((prediction - half_split) ** 2))),
        "split_half_max_abs_residual_meV": float(1000.0 * np.max(np.abs(prediction - half_split))),
        "mean_dispersion_E0_eV": float(mean_coeff[0]),
        "mean_curvature_Ca_eVA2": float(mean_coeff[1]),
        "mean_curvature_Cb_eVA2": float(mean_coeff[2]),
        "mean_dispersion_rmse_meV": float(1000.0 * np.sqrt(np.mean((mean_prediction - mean_energy) ** 2))),
        "axis_only_alpha_a_star_eVA": float(initial[0]),
        "axis_only_alpha_b_star_eVA": float(initial[1]),
        "n_points": int(len(q)),
    }


def line_fit(mapping: list[dict], selected: dict[int, dict], window: float = 0.010) -> dict:
    q, mean_energy, half_split = arrays(mapping, selected)
    qa = q[:, 0]
    use = (np.abs(qa) <= window + 1.0e-12) & (np.abs(qa) > 1.0e-12)
    x = np.abs(qa[use])
    y = half_split[use]
    alpha = float(np.dot(x, y) / np.dot(x, x))
    design = np.column_stack((np.ones(use.sum()), qa[use] ** 2))
    coeff = np.linalg.lstsq(design, mean_energy[use], rcond=None)[0]
    return {
        "window_Ainv": window,
        "n_nonzero_points": int(use.sum()),
        "alpha_a_star_eVA": alpha,
        "split_half_rmse_meV": float(1000.0 * np.sqrt(np.mean((alpha * x - y) ** 2))),
        "mean_curvature_Ca_eVA2": float(coeff[1]),
        "mean_dispersion_rmse_meV": float(1000.0 * np.sqrt(np.mean((design @ coeff - mean_energy[use]) ** 2))),
    }


def c2_symmetry_error(mapping: list[dict], selected: dict[int, dict]) -> dict:
    lookup = {
        (round(p["q_a_Ainv"], 9), round(p["q_b_Ainv"], 9)): p["global_k"]
        for p in mapping
    }
    mean_errors = []
    split_errors = []
    for p in mapping:
        partner = lookup[(round(-p["q_a_Ainv"], 9), round(p["q_b_Ainv"], 9))]
        first = selected[p["global_k"]]
        second = selected[partner]
        mean_first = 0.5 * (first["lower_energy"] + first["upper_energy"])
        mean_second = 0.5 * (second["lower_energy"] + second["upper_energy"])
        split_first = first["upper_energy"] - first["lower_energy"]
        split_second = second["upper_energy"] - second["lower_energy"]
        mean_errors.append(abs(mean_first - mean_second))
        split_errors.append(abs(split_first - split_second))
    return {
        "mapping": "C2 about real-space b: (q_a*, q_b*) -> (-q_a*, q_b*)",
        "max_mean_energy_error_meV": float(1000.0 * max(mean_errors)),
        "max_full_split_error_meV": float(1000.0 * max(split_errors)),
    }


def pt_metrics(selected: dict[int, dict]) -> dict:
    full_split = np.array(
        [1000.0 * (item["upper_energy"] - item["lower_energy"]) for item in selected.values()]
    )
    spin_sum = np.array(
        [np.linalg.norm(item["lower_spin"] + item["upper_spin"]) for item in selected.values()]
    )
    return {
        "max_full_split_meV": float(full_split.max()),
        "p95_full_split_meV": float(np.percentile(full_split, 95)),
        "max_doublet_spin_sum": float(spin_sum.max()),
        "p95_doublet_spin_sum": float(np.percentile(spin_sum, 95)),
        "status": "unresolved_at_output_precision" if full_split.max() <= RAC_RESOLUTION_MEV + 1.0e-12 else "resolved",
    }


def reciprocal_unit_vectors(supplement_root: Path, system: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation = json.loads((supplement_root / "02_kpoints_mapping" / system / "kpoint_validation.json").read_text())
    reciprocal = np.array(validation["reciprocal_rows_Ainv"], dtype=float)
    ua = reciprocal[0] / np.linalg.norm(reciprocal[0])
    ub = reciprocal[1] / np.linalg.norm(reciprocal[1])
    normal = np.cross(ua, ub)
    normal /= np.linalg.norm(normal)
    return ua, ub, normal


def write_summary_tsv(path: Path, report: dict) -> None:
    fields = [
        "system",
        "edge",
        "window_Ainv",
        "n_points",
        "alpha_a_star_eVA",
        "alpha_b_star_eVA",
        "anisotropy_alpha_a_over_b",
        "line_alpha_a_star_eVA",
        "twoD_vs_line_relative_difference_percent",
        "split_half_rmse_meV",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for system in SYSTEMS:
            for edge in EDGES:
                final = report["fits"][system][edge]["final"]
                line = report["fits"][system][edge]["gamma_x_check"]
                resolved = report["fits"][system][edge]["status"] == "resolved"
                row = {
                    "system": system,
                    "edge": edge,
                    "window_Ainv": 0.010,
                    "n_points": final["n_points"],
                    "alpha_a_star_eVA": f"{final['alpha_a_star_eVA']:.6f}" if resolved else "N/A",
                    "alpha_b_star_eVA": f"{final['alpha_b_star_eVA']:.6f}" if resolved else "N/A",
                    "anisotropy_alpha_a_over_b": f"{final['anisotropy_alpha_a_over_b']:.3f}" if resolved else "N/A",
                    "line_alpha_a_star_eVA": f"{line['alpha_a_star_eVA']:.6f}" if resolved else "N/A",
                    "twoD_vs_line_relative_difference_percent": (
                        f"{100.0 * abs(final['alpha_a_star_eVA'] - line['alpha_a_star_eVA']) / line['alpha_a_star_eVA']:.4f}"
                        if resolved
                        else "N/A"
                    ),
                    "split_half_rmse_meV": f"{final['split_half_rmse_meV']:.6f}" if resolved else "N/A",
                    "status": report["fits"][system][edge]["status"],
                }
                writer.writerow(row)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig: plt.Figure, stem: Path) -> None:
    """Save a vector PDF and a 600 dpi PNG without lossy PDF conversion."""
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



def add_polar_direction_labels(ax: plt.Axes) -> None:
    """Place reciprocal-axis labels stably outside the polar frame."""
    kw = dict(transform=ax.transAxes, fontsize=8.2, clip_on=False)
    ax.text(1.025, 0.50, r"$a^*$", ha="left", va="center", **kw)
    ax.text(0.50, 1.008, r"$b^*$", ha="center", va="bottom", **kw)
    ax.text(-0.025, 0.50, r"$-a^*$", ha="right", va="center", **kw)
    ax.text(0.50, -0.055, r"$-b^*$", ha="center", va="top", **kw)

def plot_coupling_and_stability(report: dict, output_stem: Path) -> None:
    set_plot_style()
    colors = {"S": "#2b6cb0", "Rstar": "#c53030"}
    labels = {"S": "S", "Rstar": "R*"}
    theta = np.linspace(0.0, 2.0 * np.pi, 721)

    # The bottom margin is large enough only for the polar direction labels and
    # the compact S/R* legend.  No figure-level legend is placed above panels.
    fig = plt.figure(figsize=(7.20, 2.66))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=(1.00, 1.00, 0.92),
        left=0.060, right=0.985, bottom=0.205, top=0.820,
        wspace=0.36,
    )

    polar_axes: list[plt.Axes] = []
    sr_handles = sr_labels = None

    for idx, edge in enumerate(("valence", "conduction")):
        ax = fig.add_subplot(gs[0, idx], projection="polar")
        polar_axes.append(ax)

        for system in ("S", "Rstar"):
            fit = report["fits"][system][edge]["final"]
            alpha = np.sqrt(
                (fit["alpha_a_star_eVA"] * np.cos(theta)) ** 2
                + (fit["alpha_b_star_eVA"] * np.sin(theta)) ** 2
            )
            ax.plot(
                theta, alpha,
                color=colors[system],
                linewidth=1.55,
                linestyle="-" if system == "S" else "--",
                label=labels[system],
                solid_capstyle="round",
            )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels([])
        add_polar_direction_labels(ax)

        if edge == "valence":
            ax.set_rticks((0.1, 0.3, 0.5))
            ax.set_rlim(0.0, 0.56)
            title = "(a)  Valence"
        else:
            ax.set_rticks((0.5, 1.5, 2.5))
            ax.set_rlim(0.0, 2.72)
            title = "(b)  Conduction"

        # Radial labels are placed in the upper-right quadrant, away from the
        # top b* direction label and the title.
        ax.set_rlabel_position(72)
        ax.tick_params(axis="y", labelsize=7.1, pad=0)
        ax.grid(alpha=0.32, linewidth=0.65)
        ax._panel_title = title

        if idx == 0:
            sr_handles, sr_labels = ax.get_legend_handles_labels()

    # Panel (c): relative window sensitivity.
    axc = fig.add_subplot(gs[0, 2])
    markers = {"alpha_a_star_eVA": "o", "alpha_b_star_eVA": "s"}
    line_colors = {"valence": "#2b6cb0", "conduction": "#c53030"}
    short_labels = {
        "valence_a": r"VB $a^*$", "valence_b": r"VB $b^*$",
        "conduction_a": r"CB $a^*$", "conduction_b": r"CB $b^*$",
    }
    x = np.array([81, 177, 317])
    deviations: list[float] = []

    for edge in ("valence", "conduction"):
        s_windows = report["fits"]["S"][edge]["windows"]
        r_windows = report["fits"]["Rstar"][edge]["windows"]
        for key in ("alpha_a_star_eVA", "alpha_b_star_eVA"):
            y_s = np.array([item[key] for item in s_windows])
            y_r = np.array([item[key] for item in r_windows])
            y = 0.5 * (y_s + y_r)
            deviation = 100.0 * (y - y[-1]) / y[-1]
            deviations.extend(deviation.tolist())
            axis_key = "a" if key == "alpha_a_star_eVA" else "b"
            axc.plot(
                x, deviation,
                color=line_colors[edge],
                marker=markers[key],
                linestyle="-" if axis_key == "a" else "--",
                linewidth=1.25,
                markersize=3.7,
                label=short_labels[f"{edge}_{axis_key}"],
            )

    max_dev = max(deviations)
    # All deviations are non-negative for this dataset.  The negative region is
    # intentionally reserved for a compact two-column legend, preventing any
    # overlap with the data while keeping the panel dense.
    y_top = max(0.45, max_dev + 0.055)
    y_bottom = -0.145
    axc.axhline(0.0, color="0.45", linewidth=0.75, zorder=0)
    axc.set_xlim(65, 333)
    axc.set_ylim(y_bottom, y_top)
    axc.set_xticks((81, 177, 317))
    axc.set_yticks((-0.1, 0.0, 0.1, 0.2, 0.3, 0.4))
    axc.set_xlabel("Points in radial window", labelpad=2)
    axc.set_ylabel("Deviation from 317-point fit (%)", labelpad=3)
    axc.grid(True, alpha=0.24, linewidth=0.6)
    axc.set_box_aspect(1.0)

    axc.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.018),
        bbox_transform=axc.transAxes,
        ncol=2,
        frameon=False,
        fontsize=5.95,
        handlelength=1.45,
        handletextpad=0.30,
        columnspacing=0.58,
        labelspacing=0.18,
        borderaxespad=0.0,
    )

    # Put all three panel titles on one common figure-coordinate baseline.
    # This keeps them exactly level while remaining centered over each axes.
    fig.canvas.draw()
    title_y = 0.918
    panel_titles = (
        (polar_axes[0], "(a)  Valence"),
        (polar_axes[1], "(b)  Conduction"),
        (axc, "(c)  Window sensitivity"),
    )
    for axis, text in panel_titles:
        box = axis.get_position()
        fig.text(
            0.5 * (box.x0 + box.x1), title_y, text,
            ha="center", va="center", fontsize=9.2, fontweight="normal",
        )

    # S/R* legend belongs only to panels (a,b); center it beneath those two
    # axes, close to their bottom labels.
    b0 = polar_axes[0].get_position()
    b1 = polar_axes[1].get_position()
    polar_mid_x = 0.5 * (b0.x0 + b1.x1)
    fig.legend(
        sr_handles, sr_labels,
        loc="lower center",
        bbox_to_anchor=(polar_mid_x, 0.025),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=2.15,
        handletextpad=0.55,
        columnspacing=1.25,
        borderaxespad=0.0,
    )

    save_both(fig, output_stem)

def plot_spin_texture(
    report: dict,
    maps: dict,
    selected: dict,
    supplement_root: Path,
    edge: str,
    output_stem: Path,
) -> None:
    """Plot three equal square spin-texture panels with inline panel labels."""
    set_plot_style()
    fig = plt.figure(figsize=(7.15, 2.68))
    grid = fig.add_gridspec(
        1, 4, width_ratios=(1.0, 1.0, 1.0, 0.060),
        left=0.070, right=0.915, bottom=0.180, top=0.850, wspace=0.285,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    cax = fig.add_subplot(grid[0, 3])
    systems_display = {"S": "S", "Rstar": "R*", "Rac": "Rac"}
    all_out = []
    prepared = {}
    for system in SYSTEMS:
        ua, ub, normal = reciprocal_unit_vectors(supplement_root, system)
        rows = []
        for point in maps[system]:
            if point["radius_Ainv"] > 0.010 + 1.0e-12:
                continue
            item = selected[system][edge][point["global_k"]]
            spin = (0.5 * (item["lower_spin"] + item["upper_spin"])
                    if system == "Rac" else item["lower_spin"])
            rows.append((point["q_a_Ainv"], point["q_b_Ainv"],
                         float(np.dot(spin, ua)), float(np.dot(spin, ub)),
                         float(np.dot(spin, normal))))
        prepared[system] = np.array(rows)
        if system != "Rac":
            all_out.extend(np.abs(prepared[system][:, 4]))
    vmax = max(all_out) if all_out else 1.0

    scatter = None
    for panel, (ax, system) in enumerate(zip(axes, SYSTEMS)):
        data = prepared[system]
        qa, qb, sa, sb, sn = data.T
        planar = np.hypot(sa, sb)
        keep = (planar > 2.0e-3) & (np.hypot(qa, qb) > 1.0e-12)
        scatter = ax.scatter(qa, qb, c=sn, s=9, cmap="coolwarm",
                             vmin=-vmax, vmax=vmax, linewidths=0)
        if np.any(keep):
            ax.quiver(qa[keep], qb[keep], sa[keep] / planar[keep],
                      sb[keep] / planar[keep], color="black", angles="xy",
                      scale_units="xy", scale=1250, width=0.006,
                      headwidth=3.2, headlength=4.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.0105, 0.0105)
        ax.set_ylim(-0.0105, 0.0105)
        ax.set_xticks((-0.01, 0.0, 0.01))
        ax.set_yticks((-0.01, 0.0, 0.01))
        ax.grid(alpha=0.18)
        ax.set_title(f"({chr(97 + panel)})  {systems_display[system]}",
                     pad=7, fontsize=9.0, fontweight="normal")
        ax.set_xlabel(r"$q_{a^*}$ ($\AA^{-1}$)")
        if panel == 0:
            ax.set_ylabel(r"$q_{b^*}$ ($\AA^{-1}$)")
        else:
            ax.tick_params(labelleft=False)

    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label(r"Spin normal to the $a^*$-$b^*$ plane, $\langle\sigma_n\rangle$", fontsize=7.8, labelpad=7)
    save_both(fig, output_stem)

def plot_fit_residual_map(report: dict, maps: dict, selected: dict, output_stem: Path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.2), sharex=True, sharey=True,
                             constrained_layout=True)
    scatter = None
    vmax = 0.0
    residual_sets = []
    for system in ("S", "Rstar"):
        for edge in ("valence", "conduction"):
            mapping = maps[system]
            q, _, half_split = arrays(mapping, selected[system][edge])
            use = np.linalg.norm(q, axis=1) <= 0.010 + 1.0e-12
            q = q[use]
            half_split = half_split[use]
            fit = report["fits"][system][edge]["final"]
            pred = np.sqrt((fit["alpha_a_star_eVA"] * q[:, 0]) ** 2
                           + (fit["alpha_b_star_eVA"] * q[:, 1]) ** 2)
            residual = 1000.0 * (half_split - pred)
            vmax = max(vmax, float(np.max(np.abs(residual))))
            residual_sets.append((system, edge, q, residual))
    vmax = max(vmax, 1.0e-6)
    for panel, (ax, (system, edge, q, residual)) in enumerate(zip(axes.flat, residual_sets)):
        scatter = ax.scatter(q[:, 0], q[:, 1], c=residual, s=11, cmap="coolwarm",
                             vmin=-vmax, vmax=vmax, linewidths=0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.16)
        ax.set_title(f"({chr(97 + panel)})  {system.replace('Rstar', 'R*')} {edge}",
                     fontsize=9.0, fontweight="normal", pad=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$q_{a^*}$ ($\AA^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$q_{b^*}$ ($\AA^{-1}$)")
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.85)
    colorbar.set_label(r"Half-splitting residual (meV)")
    save_both(fig, output_stem)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True, type=Path)
    parser.add_argument("--supplement-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figure_root = args.output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    merged_root = args.main_root / "01_raw_analysis_data" / "merged_selected_bands"
    maps: dict[str, dict[str, list[dict]]] = {"grid": {}, "line": {}}
    selected: dict[str, dict[str, dict[str, dict]]] = {"grid": {}, "line": {}}
    report = {
        "analysis_version": "final_c2_v1",
        "definitions": {
            "coordinate_basis": "unit reciprocal-lattice directions a* and b*; q in inverse angstrom",
            "c2_operation": "for unique real-space b axis: (q_a*,q_b*) -> (-q_a*,q_b*)",
            "mean_dispersion": "Emean=E0+Ca*q_a*^2+Cb*q_b*^2",
            "splitting": "DeltaE/2=sqrt(alpha_a*^2*q_a*^2+alpha_b*^2*q_b*^2)",
            "reported_alpha": "directional SOC coupling norm, not a signed full spin-orbit tensor",
            "final_window_Ainv": 0.010,
            "final_window_points": 317,
            "Rac_rule": "N/A when the PT-doublet splitting is at or below 0.001 meV output resolution",
            "k0_ER_rule": "not reported; model extrema lie outside the sampled line interval",
            "energy_output_quantum_eV": ENERGY_QUANTUM_EV,
        },
        "input_files": {},
        "fits": {},
        "symmetry": {"C2": {}, "Rac_PT": {}},
    }

    for mode in ("grid", "line"):
        map_name = "grid_441_map.tsv" if mode == "grid" else "line_Gamma_X_61_map.tsv"
        for system in SYSTEMS:
            map_path = args.supplement_root / "02_kpoints_mapping" / system / map_name
            band_path = merged_root / f"{system}_{mode}_merged_selected_bands.tsv.gz"
            report["input_files"][f"{system}_{mode}_map"] = {
                "archive_member": f"02_kpoints_mapping/{system}/{map_name}",
                "sha256": sha256_file(map_path),
            }
            report["input_files"][f"{system}_{mode}_bands"] = {
                "archive_member": f"01_raw_analysis_data/merged_selected_bands/{band_path.name}",
                "sha256": sha256_file(band_path),
            }
            mapping = load_map(map_path)
            bands = load_bands(band_path)
            maps[mode][system] = mapping
            selected[mode].setdefault(system, {})
            for edge in EDGES:
                selected[mode][system][edge] = select_pairs(mapping, bands, edge)

    for system in SYSTEMS:
        report["fits"].setdefault(system, {})
        report["symmetry"]["C2"].setdefault(system, {})
        report["symmetry"]["Rac_PT"].setdefault(system, {})
        for edge in EDGES:
            q_all, mean_all, half_all = arrays(maps["grid"][system], selected["grid"][system][edge])
            fits = []
            for window in WINDOWS:
                use = np.linalg.norm(q_all, axis=1) <= window + 1.0e-12
                fit = c2_fit(q_all[use], mean_all[use], half_all[use])
                fit["window_Ainv"] = window
                fits.append(fit)
            line = line_fit(maps["line"][system], selected["line"][system][edge])
            final = fits[-1]
            status = "resolved"
            if system == "Rac":
                pt = pt_metrics(selected["grid"][system][edge])
                report["symmetry"]["Rac_PT"][system][edge] = pt
                status = "unresolved_at_output_precision" if pt["status"].startswith("unresolved") else "resolved"
            report["fits"][system][edge] = {
                "status": status,
                "windows": fits,
                "final": final,
                "gamma_x_check": line,
                "independent_window_variation_percent": {
                    "alpha_a_star": float(100.0 * (max(f["alpha_a_star_eVA"] for f in fits) - min(f["alpha_a_star_eVA"] for f in fits)) / final["alpha_a_star_eVA"]) if final["alpha_a_star_eVA"] else None,
                    "alpha_b_star": float(100.0 * (max(f["alpha_b_star_eVA"] for f in fits) - min(f["alpha_b_star_eVA"] for f in fits)) / final["alpha_b_star_eVA"]) if final["alpha_b_star_eVA"] else None,
                },
            }
            report["symmetry"]["C2"][system][edge] = c2_symmetry_error(
                maps["grid"][system], selected["grid"][system][edge]
            )

    report["mirror_S_Rstar"] = {}
    for edge in EDGES:
        s = report["fits"]["S"][edge]["final"]
        r = report["fits"]["Rstar"][edge]["final"]
        report["mirror_S_Rstar"][edge] = {
            "absolute_alpha_a_difference_eVA": abs(s["alpha_a_star_eVA"] - r["alpha_a_star_eVA"]),
            "absolute_alpha_b_difference_eVA": abs(s["alpha_b_star_eVA"] - r["alpha_b_star_eVA"]),
            "relative_alpha_a_difference_percent": 100.0 * abs(s["alpha_a_star_eVA"] - r["alpha_a_star_eVA"]) / s["alpha_a_star_eVA"],
            "relative_alpha_b_difference_percent": 100.0 * abs(s["alpha_b_star_eVA"] - r["alpha_b_star_eVA"]) / s["alpha_b_star_eVA"],
        }

    (args.output_root / "final_c2_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary_tsv(args.output_root / "final_c2_summary.tsv", report)
    plot_coupling_and_stability(report, figure_root / "Fig_C2_coupling_and_stability")
    plot_spin_texture(
        report,
        maps["grid"],
        selected["grid"],
        args.supplement_root,
        "conduction",
        figure_root / "Fig_lowk_spin_texture_conduction",
    )
    plot_spin_texture(
        report,
        maps["grid"],
        selected["grid"],
        args.supplement_root,
        "valence",
        figure_root / "Fig_lowk_spin_texture_valence",
    )
    plot_fit_residual_map(report, maps["grid"], selected["grid"], figure_root / "Fig_C2_fit_residuals")
    print(json.dumps({"status": "ok", "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
