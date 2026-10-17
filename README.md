# Data and code for "Separating Global Symmetry and Local Octahedral Distortion in Rashba Spin Textures of Chiral Two-Dimensional Hybrid Perovskites"

Angran Xia and Changwei Wang

Manuscript: RSC Advances, DOI [10.1039/D6RA07234H](https://doi.org/10.1039/D6RA07234H) — accepted 29 August 2026, at proof-correction stage.

This archive contains the **processed data and post-processing code needed to reproduce the
low-*k* quantitative results and the descriptor statistics** of that paper, together with the
structural models.

It is a reproduction-verified subset assembled from the working tree. It has **not** been
byte-compared against the "Supporting Data and Code archive" that was uploaded to the journal,
so the two should not be assumed identical.

## What is here

| Directory | Contents |
|---|---|
| `code/final_c2_analysis.py` | The published C₂-constrained 2D **k·p fitting engine** that produces the directional coupling norms α_a\*, α_b\* in Table 3. |
| `code/unified_descriptors.py` | Unified periodic-neighbour structural descriptors (δ∥, δ̄_bridge, BLDI) for all nine systems. |
| `code/reproduce_final_m6_statistics.py` | Reproduces the hypothesis-family statistics (Pearson/Spearman *r*, *ρ*, *p*, BH-*q*, bootstrap CIs, LOO range) behind Figure 6 and the SI correlation table. |
| `code/final_figure_layouts.py` | Layout/assembly code for the delivered vector figures. |
| `jana_descriptor/` | Δβ_in computation (`jana_final_statistics.py`), its per-pair CSVs, and the SI consistency-check plotting scripts. |
| `descriptors/` | The authoritative CSVs the figures are drawn from, including `Figure6_authoritative_data.csv` and `nine_system_descriptor_data.csv`. |
| `structures/` | CONTCAR for all nine systems, plus the symmetrised S (P2₁), Rac (P2₁/c) and inversion-generated R\* POSCAR models. See `structures/README_structures.md`. |
| `processed_data/main/` | Band-edge energies and spin expectation values on the 317-point radial subset — the declared inputs of `final_c2_analysis.py`. |
| `processed_data/supplement/` | k-point mappings and `kpoint_validation.json` (reciprocal-lattice unit vectors) required by the fitter. |
| `processed_data/sensitivity/` | Inputs for the ENCUT = 600 eV and 4×4 k-mesh fixed-geometry sensitivity checks. |
| `processed_data/spin_arrays/`, `spin_texture/` | Spin-texture arrays. |
| `repro/` | Figure 6 regenerated from this archive, as PDF/PNG/SVG. |
| `MANIFEST_SHA256.txt` | SHA-256 and byte size of every file in this archive except itself. |

## How to reproduce

Requires Python 3 with `numpy`, `scipy`, `matplotlib`, `pandas`.

**Descriptor statistics** (runs from the repository root, no arguments):

```bash
python code/reproduce_final_m6_statistics.py
```

Verified on 16 October 2026 from a clean checkout of this repository: the script loads
`descriptors/Figure6_authoritative_data.csv` (n = 9) and prints, for the bridge-bending
hypothesis, *r* = +0.756, *p* = 0.018, *ρ* = −0.067, BH-*q* = 0.110, LOO range
[−0.055, +0.908] — matching the manuscript and Figure 6(b).

**Low-*k* coupling norms** (needs three explicit roots):

```bash
python code/final_c2_analysis.py \
    --main-root       processed_data/main \
    --supplement-root processed_data/supplement \
    --output-root     /tmp/c2_repro
```

### What this archive has been checked against

Measured on 16 October 2026, from a clean checkout of this repository, on Windows/x64 with
Python 3.13.5, numpy 2.5.3 and scipy 1.18.0 — a different environment generation from the one
used to produce the published figures:

- Running `code/final_c2_analysis.py` on the inputs shipped here writes a
  `final_c2_results.json` whose **whole-file SHA-256 is identical** to the results file used by
  the September 2026 figure-rework chain, i.e. the analysis the delivered figures and Table 3
  are drawn from:

  ```
  7c97fca1bf821482764f79173c1f6b0cc25f45e6b7701c4baf28ecb9ae58ee2f   27472 B
  ```

- All 12 input files declared by `final_c2_analysis.py` are present and match their declared
  SHA-256 values exactly.
- Every correlation coefficient printed in the manuscript, the SI and Figures 6/S4 was
  recomputed from `descriptors/` and `jana_descriptor/` and agrees to the printed precision.
- **`k₀` and `E_R` are not reported** in these results: `definitions.k0_ER_rule` states that the
  model extrema lie outside the sampled line interval. The relation α = 2E_R/k₀ is an algebraic
  identity of the parabolic-plus-linear band model (substituting E_R ≔ α²/4C and k₀ ≔ α/2C
  reduces it to α ≡ α), so any residual is floating-point round-off rather than an independent
  check. No such check is claimed here.

**Lineage.** The shipped `processed_data/results/final_c2_results.json`
(`7c97fca1bf821482764f79173c1f6b0cc25f45e6b7701c4baf28ecb9ae58ee2f`, 27,472 B) is the September
2026 lineage that Table 3 and the delivered figures are drawn from, and it is reproduced
byte-for-byte by `code/final_c2_analysis.py` run on the inputs shipped here (verified
17 October 2026, Windows/x64, Python 3.13.5). An earlier 2 August 2026 acceptance-baseline file
(26,849 B) is **not** included; it differed only in trailing floating-point digits and, in
relative terms, only on quantities that are numerically zero (the Racemate, where PT symmetry
forces α → 0).

Whole-file SHA-256 values depend on line endings. The shipped results file carries CRLF
(609 occurrences), as produced by Python on Windows; re-running the engine on a linefeed-only
platform gives identical *parsed* values but a different whole-file hash.

## What is deliberately *not* here

- **POTCAR files.** Excluded because the VASP license does not permit redistribution; the
  manuscript's Data availability statement says the same.
- **VASP input sets as run** (INCAR / KPOINTS). These are not included; the calculation
  settings are given in the manuscript's Methods and SI.
- **Raw VASP outputs** (OUTCAR / PROCAR / vasprun.xml / CHGCAR / WAVECAR). These total ~0.9 TB
  and are retained on the Shaanxi Normal University high-performance computing cluster under
  `/home/share/tcc/project/perovskite/`, covering the full opt → scf → band → spin chain for
  all nine systems. They are available from the corresponding author on reasonable request.

## Scope of the claims

- Results are 0 K, static, equilibrium-quantity calculations. No transport, finite-temperature
  or non-equilibrium property is claimed or computed here.
- The `R` structure behind the layer-resolved Figure 5 and the time-reversal Figure 7 is the
  *independently relaxed* R screening structure from the earlier PBE/PBE-D3 screening branch,
  **not** the inversion-generated R\* mirror used in the final low-*k* S/R\*/Rac analysis. The
  two are distinguished explicitly in the figure captions.
- The manuscript is at proof-correction stage; figure captions and artwork are still being
  revised with the publisher.

## Licence

The source code is licensed under the [MIT License](LICENSE). The data files,
structural models, processed numerical results, descriptor tables, and
accompanying documentation are licensed under the
[Creative Commons Attribution 4.0 International License](LICENSE-DATA).

Third-party material remains subject to its original licence. In particular,
the VASP licence does not permit redistribution of POTCAR files, so no POTCAR
files are included.
