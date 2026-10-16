#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_jana_descriptor_final.py
==================================
FINAL, self-contained reproduction of the asymmetric in-plane octahedral-tilting
disparity Delta_beta_in defined by Jana et al. (Nat. Commun. 12, 4982 (2021),
SI Note 1 / Fig. S4), evaluated on the nine PBE screening structures.

Definition implemented
----------------------
For a pair of adjacent Pb atoms sharing a bridging (equatorial) iodine, the
in-plane M-X-M angle is obtained by projecting the Pb-I-Pb geometry onto the
horizontal plane that contains the metal (Pb) sublattice atoms:

    beta_in = angle between the projections of (Pb1 - X) and (Pb2 - X)
              onto the plane fitted through the coplanar Pb sub-lattice (SVD).

Two inequivalent bridging iodines on the same Pb-Pb pair give two projected
angles, beta and beta'.  The disparity is

    Delta_beta_in = | beta - beta' | .

Per-system aggregation: beta = mean of the larger projected angle per pair,
beta' = mean of the smaller projected angle per pair, so that
beta - beta' = mean | beta - beta' | = Delta_beta_in exactly.

A second projection, onto the axial I-I direction (the two terminal iodines of
the same octahedron), is retained only as a projection-definition sensitivity
variant.  The coplanar-Pb-plane projection is the primary definition.

Inputs : the nine canonical screening CONTCAR files, either
           * a directory that contains (or is) `structures/` with CONTCAR_*, or
           * a zip archive with entries `structures/CONTCAR_*`.
         By default the script looks in ./structures, ../structures and the
         enclosing package root.
Outputs: jana_delta_beta_in_per_system_final.csv
         jana_delta_beta_in_all_pairs_final.csv

No quantity is re-extracted from the band structures; the Rashba response table is
the frozen, archived nine-system table (see the package README).

Usage:
    python reproduce_jana_descriptor_final.py [--input PATH] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------- VASP reader


def read_poscar(text: str):
    """Minimal POSCAR/CONTCAR reader (lattice, symbols, fractional coords)."""
    L = text.splitlines()
    scale = float(L[1].split()[0])
    lat = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * scale
    l5 = L[5].split()
    if not l5[0].replace('-', '').replace('.', '').isdigit():
        symbols, counts, start = l5, [int(x) for x in L[6].split()], 7
    else:
        symbols, counts, start = None, [int(x) for x in l5], 6
    n = sum(counts)
    while not L[start].strip() or L[start].strip()[0] in 'DdCcKkSs':
        start += 1
    ct = L[start - 1].strip()[0].upper()
    coords = np.array([[float(x) for x in L[start + i].split()[:3]] for i in range(n)])
    if ct == 'C':
        coords = coords @ np.linalg.inv(lat)
    symbols = (['X'] * n if symbols is None
               else list(itertools.chain.from_iterable([s] * c for s, c in zip(symbols, counts))))
    return lat, symbols, coords


def load_contcars(src: Path) -> dict:
    """Return {system_name: CONTCAR text} from a directory or a zip archive."""
    if src.is_dir():
        cand = src / 'structures' if (src / 'structures').is_dir() else src
        files = sorted(cand.glob('CONTCAR_*'))
        if not files:
            raise SystemExit(f'No CONTCAR_* files found under {cand}')
        return {p.name.split('CONTCAR_')[1]: p.read_text(errors='ignore') for p in files}
    Z = zipfile.ZipFile(src)
    names = sorted(n for n in Z.namelist() if n.startswith('structures/CONTCAR_'))
    if not names:
        raise SystemExit(f'No structures/CONTCAR_* entries in {src}')
    return {n.split('CONTCAR_')[1]: Z.read(n).decode('utf-8', 'ignore') for n in names}


def nearest_from(frac_ref, frac_all, lat, mask, k=6, rng=2):
    """Return the k nearest periodic images of masked atoms to frac_ref."""
    cf, ci = [], []
    for img in itertools.product(range(-rng, rng + 1), repeat=3):
        for i in np.nonzero(mask)[0]:
            cf.append(frac_all[i] + np.array(img))
            ci.append(i)
    cf = np.asarray(cf)
    d = np.linalg.norm((cf - frac_ref) @ lat, axis=1)
    o = np.argsort(d)[:k]
    return np.array(ci)[o], cf[o], d[o]


def angle_between(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def beta_in(pb1, pb2, x, n):
    """In-plane projected M-X-M angle.  n = plane normal (need not be unit)."""
    n = n / np.linalg.norm(n)
    v1, v2 = pb1 - x, pb2 - x
    p1 = v1 - np.dot(v1, n) * n
    p2 = v2 - np.dot(v2, n) * n
    return angle_between(p1, p2)


# ------------------------------------------------------------------ analysis


def analyse(lat, symbols, frac):
    sym = np.array(symbols)
    pb = np.nonzero(sym == 'Pb')[0]
    cart = frac @ lat
    rows = []
    for p in pb:
        idx, img_f, _ = nearest_from(frac[p], frac, lat, sym == 'I', k=6)
        i_cart = img_f @ lat
        br = []
        for j, ic in zip(idx, i_cart):
            f_i = ic @ np.linalg.inv(lat)
            m = np.ones(len(sym), bool)
            m[p] = False
            oi, of, od = nearest_from(f_i, frac, lat, m & (sym == 'Pb'), k=1)
            if od[0] > 5.0:
                continue
            oc = of[0] @ lat
            dpp = np.linalg.norm(oc - cart[p])
            if dpp > 7.0:
                continue
            br.append(dict(io=j, ipos=ic, opos=oc, dpp=dpp, dir=(oc - cart[p]) / dpp))
        if len(br) < 4:
            continue
        # axial normal: terminal (non-bridging) I-I direction of this octahedron
        bridging = set(b['io'] for b in br)
        ax = [k for k in range(6) if int(idx[k]) not in bridging]
        n_axial = (i_cart[ax[1]] - i_cart[ax[0]]) if len(ax) == 2 \
            else np.cross(br[0]['dir'], br[1]['dir'])
        # primary normal: SVD plane through this Pb + its in-plane Pb neighbours
        pts = np.vstack([cart[p]] + [b['opos'] for b in br])
        c0 = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c0)
        n_plane = vt[2]
        # group bridging iodines into antiparallel (in-plane) pairs
        used, pairs = set(), []
        for a, c in itertools.combinations(range(len(br)), 2):
            if a in used or c in used:
                continue
            if np.dot(br[a]['dir'], br[c]['dir']) < -0.7:
                pairs.append((a, c))
                used.update((a, c))
        for a, c in pairs:
            bp_a = beta_in(cart[p], br[a]['opos'], br[a]['ipos'], n_plane)
            bp_c = beta_in(cart[p], br[c]['opos'], br[c]['ipos'], n_plane)
            ba_a = beta_in(cart[p], br[a]['opos'], br[a]['ipos'], n_axial)
            ba_c = beta_in(cart[p], br[c]['opos'], br[c]['ipos'], n_axial)
            rows.append(dict(
                pb_idx=int(p),
                beta_in_plane_a=bp_a, beta_in_plane_b=bp_c,
                delta_beta_in_plane=abs(bp_a - bp_c),
                beta_in_axial_a=ba_a, beta_in_axial_b=ba_c,
                delta_beta_in_axial=abs(ba_a - ba_c)))
    return pd.DataFrame(rows)


def resolve_input(explicit):
    if explicit is not None:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for cand in [Path.cwd(), here, here / 'structures', here.parent,
                 here.parent / 'structures', here.parent.parent]:
        for c in ([cand] + ([cand / 'structures'] if cand.is_dir() else [])):
            if c.is_dir() and list(c.glob('CONTCAR_*')):
                return c
        zp = cand / '07_Supporting_Data_and_Code.zip'
        if zp.is_file():
            return zp
    raise SystemExit('Could not locate structures/CONTCAR_*; pass --input.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=None,
                    help='directory containing structures/CONTCAR_*, or a zip archive')
    ap.add_argument('--outdir', type=Path, default=None)
    a = ap.parse_args()

    src = resolve_input(a.input)
    outdir = a.outdir or Path(__file__).resolve().parent

    contcars = load_contcars(src)
    allp, persys = [], []
    for s, text in contcars.items():
        lat, symbols, frac = read_poscar(text)
        df = analyse(lat, symbols, frac)
        df.insert(0, 'system', s)
        allp.append(df)

        # oriented aggregation: beta = mean larger, beta' = mean smaller
        pl = df[['beta_in_plane_a', 'beta_in_plane_b']].to_numpy(float)
        ax = df[['beta_in_axial_a', 'beta_in_axial_b']].to_numpy(float)
        hi_p, lo_p = pl.max(1), pl.min(1)
        hi_a, lo_a = ax.max(1), ax.min(1)
        persys.append(dict(
            system=s, n_pairs=len(df),
            beta_in_plane=hi_p.mean(), beta_prime_in_plane=lo_p.mean(),
            delta_beta_in_plane=(hi_p - lo_p).mean(),
            delta_beta_in_plane_max=df.delta_beta_in_plane.max(),
            beta_in_axial=hi_a.mean(), beta_prime_in_axial=lo_a.mean(),
            delta_beta_in_axial=(hi_a - lo_a).mean()))

    allp = pd.concat(allp, ignore_index=True)
    per = pd.DataFrame(persys).sort_values('system').reset_index(drop=True)
    per['abs_proj_diff'] = (per.delta_beta_in_plane - per.delta_beta_in_axial).abs()

    outdir.mkdir(parents=True, exist_ok=True)
    allp.to_csv(outdir / 'jana_delta_beta_in_all_pairs_final.csv', index=False)
    per.to_csv(outdir / 'jana_delta_beta_in_per_system_final.csv', index=False)

    print('structures read from:', src)
    print(per.to_string(index=False))
    print('\nmax |plane - axial| = %.4f deg' % per.abs_proj_diff.max())
    print('wrote jana_delta_beta_in_per_system_final.csv, '
          'jana_delta_beta_in_all_pairs_final.csv')


if __name__ == '__main__':
    main()
