#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jana_final_statistics.py
========================
Primary statistics for the literature-defined asymmetric in-plane
octahedral-tilting disparity (Delta_beta_in) versus the branch-extremum Rashba
coefficient (alpha_R^ext = 2 E_R / k0).

The response table is the FROZEN nine-system archive table
(rashba_parameters_corrected_v2.csv); nothing is re-extracted from the bands.

Inputs (searched in the script directory and its neighbours):
  jana_delta_beta_in_per_system_final.csv
  rashba_parameters_corrected_v2.csv
Output:
  JANA_FINAL_PRIMARY_STATISTICS.csv   (tidy)

Usage:
    python jana_final_statistics.py [--per-system PATH] [--rashba PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED, NBOOT = 20260907, 100000


def find_data(name: str):
    here = Path(__file__).resolve().parent
    for cand in [here, here / 'data', here / 'descriptors',
                 here.parent, here.parent / 'descriptors',
                 here.parent / 'data', here.parent.parent,
                 here.parent.parent / 'descriptors', Path.cwd()]:
        p = cand / name
        if p.is_file():
            return p
    return None


def stats_full(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    tau, ptau = stats.kendalltau(x, y)
    loo = np.array([stats.pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(NBOOT, len(x)))
    xb, yb = x[idx], y[idx]
    xb = xb - xb.mean(1, keepdims=True)
    yb = yb - yb.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        bs = (xb * yb).sum(1) / np.sqrt((xb * xb).sum(1) * (yb * yb).sum(1))
    bs = bs[np.isfinite(bs)]
    ci = np.percentile(bs, [2.5, 97.5])
    return dict(pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=prho,
                kendall_tau=tau, kendall_p=ptau,
                loo_r_lo=loo.min(), loo_r_hi=loo.max(),
                boot_ci_lo=ci[0], boot_ci_hi=ci[1])


SUBSETS = {
    'all_n9': lambda d: d,
    'remove_Rac_n8': lambda d: d[d.system != 'Rac'],
    'remove_2BrMBA_n8': lambda d: d[d.system != '2BrMBA'],
    'remove_Rac_2BrMBA_n7': lambda d: d[~d.system.isin(['Rac', '2BrMBA'])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-system', type=Path, default=None)
    ap.add_argument('--rashba', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    per_path = a.per_system or find_data('jana_delta_beta_in_per_system_final.csv')
    rash_path = a.rashba or find_data('rashba_parameters_corrected_v2.csv')
    if per_path is None or rash_path is None:
        raise SystemExit('Could not locate the per-system or frozen response CSV; '
                         'pass --per-system / --rashba.')
    out_path = a.out or (Path(__file__).resolve().parent / 'JANA_FINAL_PRIMARY_STATISTICS.csv')

    per = pd.read_csv(per_path)
    rash = pd.read_csv(rash_path)
    d = per.merge(rash, on='system').sort_values('system').reset_index(drop=True)

    rows = []

    # ---- primary family: Delta_beta_in (Pb-plane) x alpha_R^ext -------------
    for tag, sel in SUBSETS.items():
        sub = sel(d)
        s = stats_full(sub.delta_beta_in_plane, sub.alpha_r)
        rows.append(dict(analysis='primary', descriptor='delta_beta_in_plane',
                         response='alpha_R_ext', subset=tag, n=len(sub), **s))

    # ---- BH-FDR, family m = 3 (alpha_R^ext primary; k0 / E_R components) ----
    pv = [stats_full(d.delta_beta_in_plane, d.alpha_r)['pearson_p'],
          stats_full(d.delta_beta_in_plane, d.k0)['pearson_p'],
          stats_full(d.delta_beta_in_plane, d.E_R)['pearson_p']]
    q = stats.false_discovery_control(np.array(pv), method='bh')
    for lab, p_, q_ in zip(['alpha_R_ext', 'k0_component', 'E_R_component'], pv, q):
        rows.append(dict(analysis='bh_fdr_m3', descriptor='delta_beta_in_plane',
                         response=lab, subset='all_n9', n=9,
                         pearson_p=float(p_), bh_q=float(q_)))

    # ---- components: consistency checks only (not independent findings) -----
    for col, lab in [('k0', 'k0_component'), ('E_R', 'E_R_component')]:
        for tag, sel in SUBSETS.items():
            sub = sel(d)
            s = stats_full(sub.delta_beta_in_plane, sub[col])
            rows.append(dict(analysis='consistency_component',
                             descriptor='delta_beta_in_plane', response=lab,
                             subset=tag, n=len(sub), **s))

    # ---- projection-definition sensitivity ---------------------------------
    for resp, lab in [('alpha_r', 'alpha_R_ext'), ('k0', 'k0_component'),
                      ('E_R', 'E_R_component')]:
        for tag, sel in [('all_n9', SUBSETS['all_n9']),
                         ('remove_Rac_2BrMBA_n7', SUBSETS['remove_Rac_2BrMBA_n7'])]:
            sub = sel(d)
            sp = stats_full(sub.delta_beta_in_plane, sub[resp])
            sa = stats_full(sub.delta_beta_in_axial, sub[resp])
            rows.append(dict(analysis='projection_sensitivity', descriptor='delta_beta_in_plane',
                             response=lab, subset=tag, n=len(sub), **sp))
            rows.append(dict(analysis='projection_sensitivity', descriptor='delta_beta_in_axial',
                             response=lab, subset=tag, n=len(sub), **sa))

    out = pd.DataFrame(rows)
    cols = ['analysis', 'descriptor', 'response', 'subset', 'n']
    cols += [c for c in ['pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p',
                         'kendall_tau', 'kendall_p', 'loo_r_lo', 'loo_r_hi',
                         'boot_ci_lo', 'boot_ci_hi', 'bh_q'] if c in out.columns]
    out = out[cols]
    out.to_csv(out_path, index=False, float_format='%.6g')

    print('=== merged nine-system table ===')
    print(d[['system', 'beta_in_plane', 'beta_prime_in_plane', 'delta_beta_in_plane',
             'delta_beta_in_axial', 'alpha_r', 'k0', 'E_R']].to_string(index=False))
    print()
    print('=== PRIMARY: Delta_beta_in (Pb-plane) x alpha_R_ext ===')
    print(out[out.analysis == 'primary'].to_string(index=False))
    print()
    print('wrote', out_path)


if __name__ == '__main__':
    main()
