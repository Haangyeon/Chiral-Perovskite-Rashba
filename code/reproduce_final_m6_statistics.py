#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_final_m6_statistics.py
================================
Final-release reproduction script for the manuscript's primary hypothesis family
(Table S5): n = 9 structures, m = 6 structure-to-response hypotheses
(3 descriptors x 2 responses).

This script starts from the FINAL authoritative descriptor table shipped in this
archive (`descriptors/Figure6_authoritative_data.csv`) and reproduces every number
reported in Table S5 of the Supporting Information:

    Descriptor      Response     r        p       rho      BH-q    95% CI           LOO range
    delta_par       |S|tot      -0.018   0.963   -0.017   0.963   [-0.992, 0.980]  [-0.970, 0.879]
    delta_par       |S|inorg     0.069   0.860   -0.033   0.963   [-0.984, 0.941]  [-0.047, 0.261]
    BLDI            |S|tot      -0.041   0.917    0.267   0.963   [-0.986, 0.981]  [-0.947, 0.729]
    BLDI            |S|inorg     0.074   0.850    0.300   0.963   [-0.972, 0.853]  [-0.029, 0.271]
    delta_bridge    |S|tot       0.756   0.018   -0.067   0.110   [-0.915, 0.993]  [-0.055, 0.908]
    delta_bridge    |S|inorg     0.175   0.653   -0.100   0.963   [-0.762, 0.955]  [-0.219, 0.749]

Statistical conventions (fixed; identical to those used for the published table):
  * Pearson r and its two-sided p value          (scipy.stats.pearsonr)
  * Spearman rho                                  (scipy.stats.spearmanr)
  * Benjamini-Hochberg q over the SIX tests only  (scipy.stats.false_discovery_control, method='bh')
  * Bootstrap 95% percentile CI of Pearson r      (100000 resamples, seed 20260907,
                                                   rows resampled with replacement)
  * Leave-one-out range of Pearson r              (drop each structure in turn)

NOTE ON PROVENANCE
------------------
This is a *final-release reproduction script*, written after acceptance so that the
numbers printed in Table S5 can be regenerated from the shipped data. It is not the
original exploratory script. Scripts that computed earlier, superseded descriptor
families are retained in the project's internal history but are **not** part of this
public archive.

Usage
-----
    python reproduce_final_m6_statistics.py                 # reads ./descriptors/...
    python reproduce_final_m6_statistics.py --root ..       # or an explicit archive root
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DESCRIPTORS = {
    "delta_par": "delta_inplane",       # in-plane Pb-I displacement anisotropy, delta_parallel
    "BLDI": "BLDI",                     # bond-length distortion index
    "delta_bridge": "bridge_bending",   # mean in-plane Pb-I-Pb bridge bending
}
RESPONSES = {
    "|S|tot": "mean_s_tot",
    "|S|inorg": "mean_s_inorg",
}

N_BOOTSTRAP = 100_000
BOOTSTRAP_SEED = 20_260_907

# Reference values printed in Table S5 (used for self-checking only).
REFERENCE = {
    ("delta_par", "|S|tot"):      (-0.018, 0.963, -0.017, 0.963, -0.992, 0.980, -0.970, 0.879),
    ("delta_par", "|S|inorg"):    (0.069, 0.860, -0.033, 0.963, -0.984, 0.941, -0.047, 0.261),
    ("BLDI", "|S|tot"):           (-0.041, 0.917, 0.267, 0.963, -0.986, 0.981, -0.947, 0.729),
    ("BLDI", "|S|inorg"):         (0.074, 0.850, 0.300, 0.963, -0.972, 0.853, -0.029, 0.271),
    ("delta_bridge", "|S|tot"):   (0.756, 0.018, -0.067, 0.110, -0.915, 0.993, -0.055, 0.908),
    ("delta_bridge", "|S|inorg"): (0.175, 0.653, -0.100, 0.963, -0.762, 0.955, -0.219, 0.749),
}


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile 95% CI of Pearson r by resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    xb = x[idx]
    yb = y[idx]
    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)
    num = (xb * yb).sum(axis=1)
    den = np.sqrt((xb * xb).sum(axis=1) * (yb * yb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="archive root containing descriptors/ (default: parent of this script)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the machine-comparable CSV (default: <root>/reproduced_table_S5.csv)")
    args = ap.parse_args()

    root: Path = args.root
    src = root / "descriptors" / "Figure6_authoritative_data.csv"
    if not src.exists():
        print("ERROR: expected %s" % src, file=sys.stderr)
        return 2

    df = pd.read_csv(src).sort_values("system").reset_index(drop=True)
    n = len(df)
    print("Loaded %s  (n = %d systems: %s)" % (src.name, n, ", ".join(df["system"])))
    if n != 9:
        print("WARNING: expected n = 9, found n = %d" % n, file=sys.stderr)

    # ---- point estimates -------------------------------------------------
    recs = []
    for d_lab, d_col in DESCRIPTORS.items():
        for r_lab, r_col in RESPONSES.items():
            x = df[d_col].to_numpy(float)
            y = df[r_col].to_numpy(float)
            r, p = stats.pearsonr(x, y)
            rho, _ = stats.spearmanr(x, y)
            loo = [np.corrcoef(np.delete(x, i), np.delete(y, i))[0, 1] for i in range(n)]
            ci_lo, ci_hi = _bootstrap_ci(x, y, N_BOOTSTRAP, BOOTSTRAP_SEED)
            recs.append(dict(descriptor=d_lab, response=r_lab, r=r, p=p, rho=rho,
                             ci_low=ci_lo, ci_high=ci_hi,
                             loo_min=float(min(loo)), loo_max=float(max(loo))))
    res = pd.DataFrame(recs)

    # ---- Benjamini-Hochberg over the six tests only ----------------------
    res["q"] = stats.false_discovery_control(res["p"].to_numpy(), method="bh")

    # ---- invariants ------------------------------------------------------
    tc = res[["r", "p", "rho", "q"]].to_numpy(float)
    assert np.isfinite(tc).all(), "non-finite statistic encountered"
    assert ((res["q"] >= 0) & (res["q"] <= 1)).all(), "BH q outside [0, 1]"
    assert (res["q"] >= res["p"] - 1e-12).all(), "BH q must be >= raw p for every test"
    assert res["q"].max() <= 1.0 + 1e-12
    _bq = float(res.loc[(res.descriptor == "delta_bridge") & (res.response == "|S|tot"), "q"].iloc[0])
    assert abs(_bq - 0.109903) < 5e-6, \
        "bridge-bending q does not match the published 0.109903 (got %.6f)" % _bq
    dpar_r = float(res.loc[(res.descriptor == "delta_par") & (res.response == "|S|tot"), "r"].iloc[0])
    assert abs(dpar_r - (-0.0183)) < 5e-4, "delta_par r does not match the published -0.018"

    # ---- print table S5 --------------------------------------------------
    print()
    print("Table S5 reproduction  (n = %d, m = %d structure-to-response hypotheses)"
          % (n, len(DESCRIPTORS) * len(RESPONSES)))
    print("%-14s %-9s %8s %7s %8s %7s %18s %18s" %
          ("Descriptor", "Response", "r", "p", "rho", "BH-q", "Bootstrap 95% CI", "LOO range"))
    print("-" * 100)
    for _, row in res.iterrows():
        print("%-14s %-9s %+8.3f %7.3f %+8.3f %7.3f   [%+.3f, %+.3f]   [%+.3f, %+.3f]" % (
            row.descriptor, row.response, row.r, row.p, row.rho, row.q,
            row.ci_low, row.ci_high, row.loo_min, row.loo_max))

    # ---- compare against the printed reference ---------------------------
    print()
    mism = []
    for _, row in res.iterrows():
        ref = REFERENCE[(row.descriptor, row.response)]
        checks = [
            ("r", row.r, ref[0], 5e-4), ("p", row.p, ref[1], 5e-4),
            ("rho", row.rho, ref[2], 5e-4), ("q", row.q, ref[3], 5e-4),
            ("ci_low", row.ci_low, ref[4], 1e-3), ("ci_high", row.ci_high, ref[5], 1e-3),
            ("loo_min", row.loo_min, ref[6], 1e-3), ("loo_max", row.loo_max, ref[7], 1e-3),
        ]
        for name, got, want, tol in checks:
            if abs(got - want) > tol:
                mism.append((row.descriptor, row.response, name, got, want))
    if mism:
        print("MISMATCHES vs Table S5:")
        for m in mism:
            print("   %s / %s : %s = %.6f, expected %.6f" % m)
    else:
        print("All six rows match Table S5 to the printed precision.  [OK]")

    # ---- machine-comparable output --------------------------------------
    out = args.out or (root / "reproduced_table_S5.csv")
    res.to_csv(out, index=False, float_format="%.6f")
    print("Wrote %s" % out)
    return 0 if not mism else 1


if __name__ == "__main__":
    raise SystemExit(main())
