# Data and code for "Separating Global Symmetry and Local Octahedral Distortion in Rashba Spin Textures of Chiral Two-Dimensional Hybrid Perovskites"

Angran Xia and Changwei Wang

Manuscript: RSC Advances, DOI [10.1039/D6RA07234H](https://doi.org/10.1039/D6RA07234H) — accepted 29 August 2026, at proof-correction stage.

This archive contains the **processed data and post-processing code needed to reproduce the
low-*k* quantitative results and the descriptor statistics** of that paper, together with the
structural models. It is the same material submitted to the journal as the Supporting Data and
Code archive.

## What is here

| Directory | Contents |
|---|---|
| `code/final_c2_analysis.py` | The published C₂-constrained 2D **k·p fitting engine** that produces the directional coupling norms α_a\*, α_b\* in Table 3. |
| `code/unified_descriptors.py` | Unified periodic-neighbour structural descriptors (δ∥, δ̄_bridge, BLDI) for all nine systems. |
| `code/reproduce_final_m6_statistics.py` | Reproduces the hypothesis-family statistics (Pearson/Spearman *r*, *ρ*, *p*, BH-*q*, bootstrap CIs, LOO range) behind Figure 6 and SI Table S4. |
| `code/final_figure_layouts.py` | Layout/assembly code for the delivered vector figures. |
| `jana_descriptor/` | Δβ_in computation (`jana_final_statistics.py`), its per-pair CSVs, and the SI Figure S4 plotting scripts. |
| `descriptors/` | The authoritative CSVs the figures are drawn from, including `Figure6_authoritative_data.csv` and `nine_system_descriptor_data.csv`. |
| `structures/` | POSCAR/CONTCAR for all nine systems, plus the symmetrised S (P2₁), Rac (P2₁/c) and inversion-generated R\* models. See `structures/README_structures.md`. |
| `processed_data/main/` | Band-edge energies and spin expectation values on the 317-point radial subset — the declared inputs of `final_c2_analysis.py`. |
| `processed_data/supplement/` | k-point mappings and `kpoint_validation.json` (reciprocal-lattice unit vectors) required by the fitter. |
| `processed_data/sensitivity/` | Inputs for the ENCUT = 600 eV and 4×4 k-mesh fixed-geometry sensitivity checks (SI). |
| `processed_data/spin_arrays/`, `spin_texture/` | Spin-texture arrays. |
| `repro/` | Figure 6 regenerated from this archive, as PDF/PNG/SVG. |
| `MANIFEST_SHA256.txt` | SHA-256 and byte size of every file in this archive. |

## How to reproduce

```bash
pip install numpy scipy matplotlib pandas pymatgen
python code/final_c2_analysis.py --help
python code/reproduce_final_m6_statistics.py
```

Verification performed on 16 October 2026, on a Windows/x64 host with Python 3.13.5,
numpy 2.5.3 and scipy 1.18.0 (a different environment generation from the one used for
publication):

- All 12 input files declared by `final_c2_analysis.py` were present and matched their
  declared SHA-256 values exactly.
- Re-running the published engine on those inputs reproduced **157 of 157 numeric fields
  bit-identically** against the frozen published results file.
- Every correlation coefficient printed in the manuscript, the SI and Figures 6/S4 was
  recomputed from `descriptors/` and `jana_descriptor/` and agrees to the printed precision.
- The identity α_R^ext ≡ 2E_R/k₀ holds to a maximum relative deviation of 7.95 × 10⁻¹⁴ with
  E_R expressed in meV.

## What is deliberately *not* here

- **POTCAR files.** Excluded because the VASP license does not permit redistribution. The
  pseudopotential set is documented in the Methods (PBE.64, `Pb_d`).
- **Raw VASP outputs** (OUTCAR / PROCAR / vasprun.xml / CHGCAR / WAVECAR). These total ~0.9 TB
  and are retained on the Shaanxi Normal University high-performance computing cluster under
  `/home/share/tcc/project/perovskite/`, covering the full opt → scf → band → spin chain for
  all nine systems. They are available from the corresponding author on reasonable request.
- **Input files** (INCAR / KPOINTS / POSCAR as run) are in the repository for the relaxed
  structures; the calculation settings are specified in the Methods and the SI.

## Notes

- The `R` structure used by the layer-resolved Figure 5 and the time-reversal Figure 7 is the
  *independently relaxed* R screening structure from the earlier PBE/PBE-D3 screening branch,
  **not** the inversion-generated R\* mirror used in the final low-*k* S/R\*/Rac analysis. The
  two are distinguished explicitly in the figure captions.
- Results are 0 K, static, equilibrium-quantity calculations. No transport, finite-temperature
  or non-equilibrium property is claimed or computed here.
