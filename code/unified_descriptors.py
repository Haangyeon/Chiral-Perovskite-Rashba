#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Structural Descriptor Calculator for 2D Hybrid Perovskites
======================================================================
Implements the descriptor definitions specified by Astra (ChatGPT 6)
for the chiral perovskite CISS project. All definitions are formula-precise
and reproducible.

Descriptors implemented:
  1. Six-nearest iodine coordination (periodic, distance-sorted)
  2. Bond Length Distortion Index (BLDI) — dimensionless
  3. Pb off-centering displacement (in-plane / out-of-plane / total)
  4. Pb-I-Pb bridge angles (mean bending, direction anisotropy)
  5. Space group verification (spglib, multi-symprec scan)

Usage:
  python unified_descriptors.py --contcar <path> [--output <csv>]
  python unified_descriptors.py --batch <dir_containing_opt_*> [--output <csv>]
  python unified_descriptors.py --test  (run unit tests)

Author: perovskite project (proof-stage audit)
Date: 2026-09-07
"""

import os
import sys
import csv
import hashlib
import argparse
import numpy as np
from numpy.linalg import svd, norm

# ---------------------------------------------------------------------------
# 1. CONTCAR Parser (pure Python, no pymatgen dependency)
# ---------------------------------------------------------------------------

def read_contcar(filepath):
    """
    Parse a VASP CONTCAR/POSCAR file.

    Returns:
        dict with keys:
            comment (str), scale (float), lattice (3x3 ndarray),
            species (list of str), ntypes (list of int),
            coords (Nx3 ndarray, Cartesian), selective (bool),
            coord_type (str: 'Direct' or 'Cartesian')
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    idx = 0
    comment = lines[idx].strip(); idx += 1
    scale = float(lines[idx].split()[0]); idx += 1

    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
    lattice *= scale

    # Species line (may be absent in older format)
    species_tokens = lines[idx].split()
    if all(t.isalpha() for t in species_tokens):
        species = species_tokens
        idx += 1
        ntypes = [int(x) for x in lines[idx].split()]
        idx += 1
    else:
        ntypes = [int(x) for x in species_tokens]
        idx += 1
        species = [f'elem{i}' for i in range(len(ntypes))]

    natoms = sum(ntypes)

    # Selective dynamics?
    selective = False
    if lines[idx].strip().lower().startswith('s'):
        selective = True
        idx += 1

    coord_type = lines[idx].strip().lower(); idx += 1
    is_direct = coord_type.startswith('d')

    coords = np.zeros((natoms, 3))
    for i in range(natoms):
        coords[i] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1

    if is_direct:
        coords = coords @ lattice  # fractional -> Cartesian

    # Expand species list
    species_full = []
    for sp, n in zip(species, ntypes):
        species_full.extend([sp] * n)

    return {
        'comment': comment,
        'scale': scale,
        'lattice': lattice,
        'species': species_full,
        'ntypes': ntypes,
        'coords': coords,
        'selective': selective,
        'coord_type': 'Direct' if is_direct else 'Cartesian',
        'natoms': natoms,
    }


# ---------------------------------------------------------------------------
# 2. Periodic Neighbor Finding
# ---------------------------------------------------------------------------

def _minimum_image(delta, lattice):
    """Apply minimum image convention to a displacement vector."""
    # Convert to fractional
    inv_lat = np.linalg.inv(lattice)
    frac = delta @ inv_lat
    # Wrap to [-0.5, 0.5]
    frac = frac - np.round(frac)
    return frac @ lattice


def find_nearest_neighbors(center_idx, coords, species, lattice,
                            target_species='I', n_neighbors=6,
                            n_supercell=1):
    """
    Find the n nearest neighbors of target_species around center_idx,
    considering periodic boundary conditions.

    Args:
        center_idx: index of the center atom
        coords: Nx3 Cartesian coordinates
        species: list of species strings
        lattice: 3x3 lattice matrix
        target_species: species to search for (default 'I')
        n_neighbors: number of nearest neighbors to return
        n_supercell: supercell expansion range (default 1 = -1..1 in each dir)

    Returns:
        dict with keys:
            distances (ndarray, sorted),
            indices (ndarray, original atom indices),
            images (ndarray, Nx3 lattice translation vectors),
            displacements (ndarray, Nx3 from center to neighbor)
    """
    center = coords[center_idx]
    target_indices = [i for i, s in enumerate(species) if s == target_species]

    all_distances = []
    all_displacements = []
    all_indices = []
    all_images = []

    ranges = range(-n_supercell, n_supercell + 1)
    for ti in target_indices:
        for a in ranges:
            for b in ranges:
                for c in ranges:
                    image = np.array([a, b, c], dtype=float) @ lattice
                    neighbor_pos = coords[ti] + image
                    disp = neighbor_pos - center
                    d = norm(disp)
                    if d > 1e-10:  # exclude self
                        all_distances.append(d)
                        all_displacements.append(disp)
                        all_indices.append(ti)
                        all_images.append(np.array([a, b, c]))

    all_distances = np.array(all_distances)
    all_displacements = np.array(all_displacements)
    all_indices = np.array(all_indices)
    all_images = np.array(all_images)

    # Sort by distance
    sort_idx = np.argsort(all_distances)
    return {
        'distances': all_distances[sort_idx][:n_neighbors],
        'indices': all_indices[sort_idx][:n_neighbors],
        'images': all_images[sort_idx][:n_neighbors],
        'displacements': all_displacements[sort_idx][:n_neighbors],
    }


def find_nearest_from_position(center_pos, coords, species, lattice,
                                target_species='I', n_neighbors=6, n_supercell=2):
    """
    Find the n nearest neighbors of target_species around an arbitrary position,
    considering periodic boundary conditions.

    Args:
        center_pos: 3-vector, the center position (Cartesian)
        coords, species, lattice: structure data
        target_species: species to search for
        n_neighbors: number of nearest neighbors
        n_supercell: supercell expansion range

    Returns:
        dict with keys: distances, indices, images, displacements
    """
    target_indices = [i for i, s in enumerate(species) if s == target_species]
    all_distances = []
    all_displacements = []
    all_indices = []
    all_images = []

    ranges = range(-n_supercell, n_supercell + 1)
    for ti in target_indices:
        for a in ranges:
            for b in ranges:
                for c in ranges:
                    image = np.array([a, b, c], dtype=float) @ lattice
                    neighbor_pos = coords[ti] + image
                    disp = neighbor_pos - center_pos
                    d = norm(disp)
                    if d > 1e-10:
                        all_distances.append(d)
                        all_displacements.append(disp)
                        all_indices.append(ti)
                        all_images.append(np.array([a, b, c]))

    all_distances = np.array(all_distances)
    all_displacements = np.array(all_displacements)
    all_indices = np.array(all_indices)
    all_images = np.array(all_images)

    sort_idx = np.argsort(all_distances)
    return {
        'distances': all_distances[sort_idx][:n_neighbors],
        'indices': all_indices[sort_idx][:n_neighbors],
        'images': all_images[sort_idx][:n_neighbors],
        'displacements': all_displacements[sort_idx][:n_neighbors],
    }


def find_nearest_pb_neighbors(center_idx, coords, species, lattice,
                               n_neighbors=4, n_supercell=2):
    """
    Find the n nearest Pb neighbors (excluding self) for layer normal
    determination via SVD.
    """
    return find_nearest_neighbors(center_idx, coords, species, lattice,
                                  target_species='Pb', n_neighbors=n_neighbors,
                                  n_supercell=n_supercell)


# ---------------------------------------------------------------------------
# 3. BLDI (Bond Length Distortion Index)
# ---------------------------------------------------------------------------

def calculate_bldi(pb_idx, coords, species, lattice, n_i=6):
    """
    Calculate the Bond Length Distortion Index (BLDI) for a Pb site.

    BLDI = (1/n) * Σ |d_i - d_mean| / d_mean

    This is a dimensionless quantity.

    Args:
        pb_idx: index of Pb atom
        coords, species, lattice: structure data
        n_i: number of iodine neighbors (default 6)

    Returns:
        dict with keys: bldi (float), distances (ndarray), d_mean (float),
                        d6_d5 (float), d7_d6 (float), warnings (list)
    """
    # Get 7 neighbors to compute d6/d5 and d7/d6 ratios
    nn = find_nearest_neighbors(pb_idx, coords, species, lattice,
                                 target_species='I', n_neighbors=7)
    distances = nn['distances']
    d6 = distances[:6]
    d_mean = np.mean(d6)
    bldi = np.mean(np.abs(d6 - d_mean)) / d_mean

    warnings = []
    d6_d5 = distances[5] / distances[4] if len(distances) >= 6 else None
    d7_d6 = distances[6] / distances[5] if len(distances) >= 7 else None

    if distances[5] > 3.5:
        warnings.append(f'long 6th contact: d6={distances[5]:.4f} A > 3.5 A')

    return {
        'bldi': bldi,
        'distances': d6,
        'd_mean': d_mean,
        'd6_d5': d6_d5,
        'd7_d6': d7_d6,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# 4. Pb Off-Centering Displacement
# ---------------------------------------------------------------------------

def determine_layer_normal(pb_idx, coords, species, lattice, n_pb=4):
    """
    Determine the inorganic layer normal via SVD of the nearest Pb-Pb
    displacement vectors.

    Procedure (per Astra):
      1. For each Pb, enumerate periodic Pb images, select nearest n_pb
         non-self Pb neighbors.
      2. Check that there is a clear gap between 4th and 5th Pb neighbor.
      3. Collect all 4 Pb-Pb displacement vectors.
      4. SVD/PCA: the smallest singular value direction is the layer normal.

    Returns:
        dict with keys: normal (3-vector), gap_ratio (float), warnings (list)
    """
    # Get 5 neighbors to check gap
    nn = find_nearest_pb_neighbors(pb_idx, coords, species, lattice,
                                    n_neighbors=5)
    distances = nn['distances']
    displacements = nn['displacements']

    warnings = []
    if len(distances) < 5:
        warnings.append(f'only {len(distances)} Pb neighbors found')
        gap_ratio = None
    else:
        gap_ratio = distances[4] / distances[3]
        if gap_ratio < 1.1:
            warnings.append(f'4th-5th Pb gap small: ratio={gap_ratio:.3f}')

    # Use first 4 displacement vectors for SVD
    vecs = displacements[:4]
    # Center the vectors (subtract mean)
    vecs_centered = vecs - np.mean(vecs, axis=0)
    # SVD
    U, S, Vt = svd(vecs_centered, full_matrices=False)
    # Smallest singular value direction = layer normal
    normal = Vt[-1]
    # Ensure normal points in +z direction (arbitrary convention)
    if normal[2] < 0:
        normal = -normal

    return {
        'normal': normal,
        'gap_ratio': gap_ratio,
        'warnings': warnings,
    }


def calculate_pb_displacement(pb_idx, coords, species, lattice):
    """
    Calculate Pb off-centering displacement relative to the geometric
    centroid of its six nearest iodine neighbors.

    Definitions (per Astra):
      r_I,c = (1/6) Σ r_i  (geometric centroid, simple arithmetic mean)
      Δr_Pb = r_Pb - r_I,c
      δ_⊥ = |Δr_Pb · n̂|  (out-of-plane)
      δ_∥ = ||Δr_Pb - (Δr_Pb · n̂) n̂||  (in-plane)
      δ_tot = ||Δr_Pb||  (total)

    Layer normal n̂ is determined via SVD of nearest Pb-Pb vectors.

    Returns:
        dict with keys: delta_inplane, delta_outofplane, delta_total,
                        centroid (3-vector), normal (3-vector), warnings (list)
    """
    # Get six nearest I
    nn_i = find_nearest_neighbors(pb_idx, coords, species, lattice,
                                   target_species='I', n_neighbors=6)
    # I positions (with periodic images)
    i_positions = coords[nn_i['indices']] + nn_i['images'] @ lattice
    centroid = np.mean(i_positions, axis=0)

    # Pb position
    pb_pos = coords[pb_idx]
    delta = pb_pos - centroid

    # Layer normal
    ln = determine_layer_normal(pb_idx, coords, species, lattice)
    normal = ln['normal']

    # Decompose
    delta_outofplane = np.abs(np.dot(delta, normal))
    delta_inplane = norm(delta - np.dot(delta, normal) * normal)
    delta_total = norm(delta)

    warnings = list(ln['warnings'])

    return {
        'delta_inplane': delta_inplane,
        'delta_outofplane': delta_outofplane,
        'delta_total': delta_total,
        'centroid': centroid,
        'normal': normal,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# 5. Pb-I-Pb Bridge Angles
# ---------------------------------------------------------------------------

def calculate_bridge_angles(pb_idx, coords, species, lattice, all_pb_indices):
    """
    Calculate Pb-I-Pb bridge angles and direction anisotropy for a Pb site.

    Procedure (per Astra):
      1. Build Pb-I coordination graph using six-nearest relation.
      2. If the same periodic I belongs to two Pb's six-neighbor sets,
         it forms a Pb-I-Pb bridge.
      3. Bridge angle: theta = cos^-1((Pb1-I).(Pb2-I)/(|Pb1-I||Pb2-I|))
      4. For each Pb's four in-plane Pb neighbors, group into two
         anti-parallel direction pairs (minimize anti-parallel deviation).
      5. Mean bridge bending: delta_bridge = 180 - (theta_u + theta_v)/2
      6. Direction anisotropy: A_bridge = |theta_u - theta_v|

    Note: in-plane Pb neighbors are periodic images, not necessarily
    the other Pb in the unit cell (which may be in a different layer).

    Returns:
        dict with keys: mean_bridge_bending, direction_anisotropy,
                        effective_tilt_anisotropy, bridge_angles (list),
                        direction_groups (dict), warnings (list)
    """
    warnings = []
    pb_pos = coords[pb_idx]

    # Get this Pb's six nearest I (with images)
    nn_i = find_nearest_neighbors(pb_idx, coords, species, lattice,
                                   target_species='I', n_neighbors=6, n_supercell=2)
    # Store actual positions of my six nearest I (for matching by position)
    my_i_positions = []
    for k in range(6):
        idx = nn_i['indices'][k]
        img = nn_i['images'][k]
        actual_pos = coords[idx] + img @ lattice
        my_i_positions.append(actual_pos)

    # Find nearest Pb neighbors (considering periodic images)
    # We need enough candidates to find 4 in-plane bridges
    nn_pb = find_nearest_neighbors(pb_idx, coords, species, lattice,
                                    target_species='Pb', n_neighbors=12, n_supercell=2)

    # Determine layer normal for identifying in-plane neighbors
    layer_normal = determine_layer_normal(pb_idx, coords, species, lattice)['normal']

    # Find bridges: for each Pb neighbor (periodic image), check if any of
    # its 6 nearest I is at the same physical position as one of mine
    bridges = []  # (other_pb_actual_pos, i_pos, angle, inplane_direction)
    pos_tolerance = 0.1  # Angstrom tolerance for position matching

    for k in range(len(nn_pb['distances'])):
        other_pb_idx = nn_pb['indices'][k]
        other_img = nn_pb['images'][k]
        other_pb_pos = coords[other_pb_idx] + other_img @ lattice

        # Check if this is an in-plane neighbor (z-displacement small)
        pb_pb_disp = other_pb_pos - pb_pos
        out_of_plane = abs(np.dot(pb_pb_disp, layer_normal))
        if out_of_plane > 3.0:  # skip inter-layer neighbors
            continue

        # Get six nearest I for this Pb neighbor (using its actual position)
        nn_other = find_nearest_from_position(other_pb_pos, coords, species, lattice,
                                               target_species='I', n_neighbors=6, n_supercell=2)
        found_bridge = False
        for ki in range(6):
            oidx = nn_other['indices'][ki]
            oimg = nn_other['images'][ki]
            other_i_pos = coords[oidx] + oimg @ lattice
            # Check if this I position matches any of my six nearest I
            for mi, my_pos in enumerate(my_i_positions):
                dist = norm(other_i_pos - my_pos)
                if dist < pos_tolerance:
                    # Found a bridge! Calculate angle
                    i_pos = my_pos
                    v1 = pb_pos - i_pos
                    v2 = other_pb_pos - i_pos
                    cos_theta = np.dot(v1, v2) / (norm(v1) * norm(v2))
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))
                    # In-plane direction
                    inplane_disp = pb_pb_disp - np.dot(pb_pb_disp, layer_normal) * layer_normal
                    if norm(inplane_disp) > 1e-10:
                        direction = inplane_disp / norm(inplane_disp)
                    else:
                        direction = np.array([1, 0, 0])
                    bridges.append({
                        'other_pb_pos': other_pb_pos,
                        'i_pos': i_pos,
                        'angle': angle,
                        'direction': direction,
                        'pb_pb_distance': norm(pb_pb_disp),
                    })
                    found_bridge = True
                    break
            if found_bridge:
                break  # one bridge per Pb neighbor is enough

    if len(bridges) < 4:
        warnings.append(f'only {len(bridges)} bridges found (need >=4)')
        return {
            'mean_bridge_bending': None,
            'direction_anisotropy': None,
            'effective_tilt_anisotropy': None,
            'bridge_angles': [b['angle'] for b in bridges],
            'direction_groups': {},
            'warnings': warnings,
        }

    # Use first 4 bridges (closest in-plane Pb neighbors)
    bridges = bridges[:4]
    bridge_data = bridges

    # Group into two anti-parallel pairs
    # Strategy: find the pair with highest anti-parallel correlation
    n = len(bridge_data)
    best_pairs = None
    best_score = -np.inf

    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j are anti-parallel
            dot_ij = np.dot(bridge_data[i]['direction'], bridge_data[j]['direction'])
            if dot_ij < -0.5:  # roughly anti-parallel
                # Find the other pair
                remaining = [k for k in range(n) if k != i and k != j]
                if len(remaining) >= 2:
                    for k in range(len(remaining)):
                        for l in range(k + 1, len(remaining)):
                            rk, rl = remaining[k], remaining[l]
                            dot_kl = np.dot(bridge_data[rk]['direction'], bridge_data[rl]['direction'])
                            if dot_kl < -0.5:
                                score = abs(dot_ij) + abs(dot_kl)
                                if score > best_score:
                                    best_score = score
                                    best_pairs = ([i, j], [rk, rl])

    if best_pairs is None:
        warnings.append('could not form two anti-parallel direction pairs')
        return {
            'mean_bridge_bending': None,
            'direction_anisotropy': None,
            'effective_tilt_anisotropy': None,
            'bridge_angles': [b['angle'] for b in bridge_data],
            'direction_groups': {},
            'warnings': warnings,
        }

    pair_u, pair_v = best_pairs
    theta_u = np.mean([bridge_data[k]['angle'] for k in pair_u])
    theta_v = np.mean([bridge_data[k]['angle'] for k in pair_v])

    mean_bridge_bending = 180.0 - (theta_u + theta_v) / 2.0
    direction_anisotropy = abs(theta_u - theta_v)
    effective_tilt_anisotropy = direction_anisotropy / 2.0

    return {
        'mean_bridge_bending': mean_bridge_bending,
        'direction_anisotropy': direction_anisotropy,
        'effective_tilt_anisotropy': effective_tilt_anisotropy,
        'bridge_angles': [b['angle'] for b in bridge_data],
        'theta_u': theta_u,
        'theta_v': theta_v,
        'direction_groups': {'pair_u': pair_u, 'pair_v': pair_v},
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# 6. Space Group Check (spglib)
# ---------------------------------------------------------------------------

def check_space_group(contcar_data, symprec_list=None):
    """
    Check space group using spglib at multiple symprec values.

    Per Astra: use spglib==2.7.0, scan symprec from 1e-5 to 1e-2,
    report international symbol, number, Hall symbol, point group,
    number of symmetry operations, and max Cartesian mapping residual.

    Nominal reporting threshold: 1e-3 A.
    P2_1 (#4) must be stable from 1e-4 to 3e-3 to be formally assigned.
    """
    try:
        import spglib
    except ImportError:
        return {'error': 'spglib not installed', 'spglib_version': None}

    if symprec_list is None:
        symprec_list = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    lattice = contcar_data['lattice']
    species = contcar_data['species']
    coords = contcar_data['coords']

    # Convert to fractional coordinates for spglib
    inv_lat = np.linalg.inv(lattice)
    frac_coords = coords @ inv_lat

    # Map species to atomic numbers
    from collections import defaultdict
    unique_species = list(dict.fromkeys(species))
    species_to_num = {}
    # Simple atomic number mapping for common elements
    atomic_numbers = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
        'K': 19, 'Ca': 20, 'Fe': 26, 'Cu': 29, 'Zn': 30,
        'Br': 35, 'I': 53, 'Pb': 82, 'Cs': 55, 'Ba': 56, 'Sn': 50,
    }
    for sp in unique_species:
        species_to_num[sp] = atomic_numbers.get(sp, 0)

    numbers = [species_to_num[sp] for sp in species]

    cell = (lattice, frac_coords, numbers)

    results = {}
    for symprec in symprec_list:
        try:
            dataset = spglib.get_symmetry_dataset(
                cell, symprec=symprec, angle_tolerance=-1.0
            )
            if dataset is not None:
                results[symprec] = {
                    'international': dataset.international,
                    'number': dataset.number,
                    'hall': dataset.hall,
                    'hall_number': dataset.hall_number,
                    'pointgroup': dataset.pointgroup,
                    'n_operations': len(dataset.rotations),
                }
            else:
                results[symprec] = {'error': 'dataset is None'}
        except Exception as e:
            results[symprec] = {'error': str(e)}

    # Determine stability of P2_1
    p21_stable_range = []
    for symprec in symprec_list:
        r = results.get(symprec, {})
        if r.get('number') == 4:
            p21_stable_range.append(symprec)

    # Nominal result at 1e-3
    nominal = results.get(1e-3, {})

    return {
        'spglib_version': spglib.__version__,
        'nominal_symprec': 1e-3,
        'nominal': nominal,
        'p21_stable_range': p21_stable_range,
        'all_results': results,
    }


# ---------------------------------------------------------------------------
# 7. System-Level Processing
# ---------------------------------------------------------------------------

def process_system(contcar_path, system_name=None):
    """
    Process a single CONTCAR file and compute all descriptors.

    Returns:
        dict with system-level descriptor statistics
    """
    if system_name is None:
        system_name = os.path.basename(os.path.dirname(contcar_path))

    struct = read_contcar(contcar_path)
    species = struct['species']
    coords = struct['coords']
    lattice = struct['lattice']

    pb_indices = [i for i, s in enumerate(species) if s == 'Pb']

    # Per-Pb descriptors
    bldi_list = []
    delta_inplane_list = []
    delta_outofplane_list = []
    delta_total_list = []
    bridge_bending_list = []
    bridge_anisotropy_list = []
    all_warnings = []

    for pb_idx in pb_indices:
        # BLDI
        bldi_result = calculate_bldi(pb_idx, coords, species, lattice)
        bldi_list.append(bldi_result['bldi'])
        all_warnings.extend([f'Pb{pb_idx}: {w}' for w in bldi_result['warnings']])

        # Pb displacement
        disp_result = calculate_pb_displacement(pb_idx, coords, species, lattice)
        delta_inplane_list.append(disp_result['delta_inplane'])
        delta_outofplane_list.append(disp_result['delta_outofplane'])
        delta_total_list.append(disp_result['delta_total'])
        all_warnings.extend([f'Pb{pb_idx}: {w}' for w in disp_result['warnings']])

        # Bridge angles
        bridge_result = calculate_bridge_angles(pb_idx, coords, species, lattice, pb_indices)
        if bridge_result['mean_bridge_bending'] is not None:
            bridge_bending_list.append(bridge_result['mean_bridge_bending'])
            bridge_anisotropy_list.append(bridge_result['direction_anisotropy'])
        all_warnings.extend([f'Pb{pb_idx}: {w}' for w in bridge_result['warnings']])

    # System-level statistics
    result = {
        'system': system_name,
        'contcar_path': contcar_path,
        'n_pb': len(pb_indices),
        'volume': abs(np.linalg.det(lattice)),
        # BLDI
        'bldi_mean': np.mean(bldi_list) if bldi_list else None,
        'bldi_std': np.std(bldi_list) if bldi_list else None,
        'bldi_min': np.min(bldi_list) if bldi_list else None,
        'bldi_max': np.max(bldi_list) if bldi_list else None,
        # Pb displacement
        'delta_inplane_mean': np.mean(delta_inplane_list) if delta_inplane_list else None,
        'delta_outofplane_mean': np.mean(delta_outofplane_list) if delta_outofplane_list else None,
        'delta_total_mean': np.mean(delta_total_list) if delta_total_list else None,
        # Bridge angles
        'bridge_bending_mean': np.mean(bridge_bending_list) if bridge_bending_list else None,
        'bridge_anisotropy_mean': np.mean(bridge_anisotropy_list) if bridge_anisotropy_list else None,
        # Warnings
        'warnings': all_warnings,
    }

    # SHA256 of CONTCAR
    with open(contcar_path, 'rb') as f:
        result['contcar_sha256'] = hashlib.sha256(f.read()).hexdigest()

    return result


# ---------------------------------------------------------------------------
# 8. Batch Processing
# ---------------------------------------------------------------------------

def process_batch(base_dir, output_csv=None):
    """
    Process all opt_* directories under base_dir.

    Expected structure: base_dir/opt_<system>/CONTCAR
    """
    systems = []
    for d in sorted(os.listdir(base_dir)):
        if d.startswith('opt_') and os.path.isdir(os.path.join(base_dir, d)):
            contcar = os.path.join(base_dir, d, 'CONTCAR')
            if os.path.exists(contcar):
                system_name = d.replace('opt_', '')
                systems.append((system_name, contcar))

    results = []
    for name, path in systems:
        print(f'Processing {name}...', flush=True)
        try:
            r = process_system(path, system_name=name)
            results.append(r)
            bb = r["bridge_bending_mean"]
            bb_str = f'{bb:.4f}' if bb is not None else 'N/A'
            print(f'  BLDI={r["bldi_mean"]:.6f}, '
                  f'delta_inplane={r["delta_inplane_mean"]:.6f}, '
                  f'delta_total={r["delta_total_mean"]:.6f}, '
                  f'bridge_bending={bb_str}', flush=True)
            if r['warnings']:
                for w in r['warnings'][:3]:
                    print(f'  WARNING: {w}', flush=True)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            results.append({'system': name, 'error': str(e)})

    if output_csv:
        write_results_csv(results, output_csv)
        print(f'\nResults written to {output_csv}', flush=True)

    return results


def write_results_csv(results, output_path):
    """Write descriptor results to CSV."""
    fieldnames = [
        'system', 'n_pb', 'volume',
        'bldi_mean', 'bldi_std', 'bldi_min', 'bldi_max',
        'delta_inplane_mean', 'delta_outofplane_mean', 'delta_total_mean',
        'bridge_bending_mean', 'bridge_anisotropy_mean',
        'contcar_sha256', 'warnings',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = dict(r)
            row['warnings'] = '; '.join(r.get('warnings', []))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 9. Unit Tests
# ---------------------------------------------------------------------------

def run_tests():
    """Run unit tests for the descriptor calculator."""
    print('=' * 60)
    print('UNIT TESTS')
    print('=' * 60)

    passed = 0
    failed = 0

    def test(name, condition, detail=''):
        nonlocal passed, failed
        if condition:
            print(f'  PASS: {name}')
            passed += 1
        else:
            print(f'  FAIL: {name} {detail}')
            failed += 1

    # Test 1: CONTCAR parsing
    print('\n--- Test 1: CONTCAR Parsing ---')
    test_contcar = os.path.join(os.path.dirname(__file__), '..', '03_vasp_current', 'opt', 'opt_S', 'CONTCAR')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        test('natoms == 94', struct['natoms'] == 94, f'got {struct["natoms"]}')
        test('lattice shape 3x3', struct['lattice'].shape == (3, 3))
        test('species count matches', len(struct['species']) == struct['natoms'])
        test('volume > 0', np.linalg.det(struct['lattice']) > 0)
        print(f'  Volume: {abs(np.linalg.det(struct["lattice"])):.2f} A^3')
        print(f'  Species: {dict.fromkeys(struct["species"])}')
    else:
        print(f'  SKIP: test CONTCAR not found at {test_contcar}')

    # Test 2: Periodic neighbor finding
    print('\n--- Test 2: Periodic Neighbor Finding ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        pb_idx = next(i for i, s in enumerate(struct['species']) if s == 'Pb')
        nn = find_nearest_neighbors(pb_idx, struct['coords'], struct['species'],
                                     struct['lattice'], target_species='I', n_neighbors=6)
        test('6 neighbors found', len(nn['distances']) == 6)
        test('distances sorted', all(nn['distances'][i] <= nn['distances'][i+1] for i in range(5)))
        test('all distances > 2.5 A (reasonable Pb-I)', all(d > 2.5 for d in nn['distances']),
             f'min={nn["distances"][0]:.3f}')
        test('all distances < 4.5 A', all(d < 4.5 for d in nn['distances']),
             f'max={nn["distances"][-1]:.3f}')
        print(f'  Pb-I distances: {nn["distances"]}')

    # Test 3: BLDI
    print('\n--- Test 3: BLDI ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        pb_idx = next(i for i, s in enumerate(struct['species']) if s == 'Pb')
        bldi = calculate_bldi(pb_idx, struct['coords'], struct['species'], struct['lattice'])
        test('BLDI >= 0', bldi['bldi'] >= 0)
        test('BLDI < 0.5 (reasonable)', bldi['bldi'] < 0.5, f'got {bldi["bldi"]:.6f}')
        test('d_mean > 2.8', bldi['d_mean'] > 2.8)
        print(f'  BLDI = {bldi["bldi"]:.6f}')
        print(f'  d_mean = {bldi["d_mean"]:.4f} A')

    # Test 4: Pb displacement
    print('\n--- Test 4: Pb Off-Centering Displacement ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        pb_idx = next(i for i, s in enumerate(struct['species']) if s == 'Pb')
        disp = calculate_pb_displacement(pb_idx, struct['coords'], struct['species'], struct['lattice'])
        test('delta_total >= delta_inplane', disp['delta_total'] >= disp['delta_inplane'] - 1e-10)
        test('delta_total >= delta_outofplane', disp['delta_total'] >= disp['delta_outofplane'] - 1e-10)
        test('normal is unit vector', abs(norm(disp['normal']) - 1.0) < 1e-10)
        test('delta_total < 1.0 A (reasonable)', disp['delta_total'] < 1.0,
             f'got {disp["delta_total"]:.6f}')
        print(f'  delta_inplane = {disp["delta_inplane"]:.6f} A')
        print(f'  delta_outofplane = {disp["delta_outofplane"]:.6f} A')
        print(f'  delta_total = {disp["delta_total"]:.6f} A')
        print(f'  layer normal = {disp["normal"]}')

    # Test 5: Layer normal determination
    print('\n--- Test 5: Layer Normal Determination ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        pb_idx = next(i for i, s in enumerate(struct['species']) if s == 'Pb')
        ln = determine_layer_normal(pb_idx, struct['coords'], struct['species'], struct['lattice'])
        test('normal is unit vector', abs(norm(ln['normal']) - 1.0) < 1e-10)
        test('normal z-component significant (2D material)', abs(ln['normal'][2]) > 0.3,
             f'z={ln["normal"][2]:.4f}')
        print(f'  normal = {ln["normal"]}')
        print(f'  4th-5th Pb gap ratio = {ln["gap_ratio"]}')

    # Test 6: Bridge angles
    print('\n--- Test 6: Pb-I-Pb Bridge Angles ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        pb_indices = [i for i, s in enumerate(struct['species']) if s == 'Pb']
        pb_idx = pb_indices[0]
        ba = calculate_bridge_angles(pb_idx, struct['coords'], struct['species'],
                                      struct['lattice'], pb_indices)
        if ba['mean_bridge_bending'] is not None:
            test('bridge bending between 0 and 90 deg',
                 0 <= ba['mean_bridge_bending'] <= 90,
                 f'got {ba["mean_bridge_bending"]:.2f}')
            test('direction anisotropy >= 0', ba['direction_anisotropy'] >= 0)
            print(f'  mean bridge bending = {ba["mean_bridge_bending"]:.4f} deg')
            print(f'  direction anisotropy = {ba["direction_anisotropy"]:.4f} deg')
            print(f'  effective tilt anisotropy = {ba["effective_tilt_anisotropy"]:.4f} deg')
            print(f'  bridge angles = {ba["bridge_angles"]}')
        else:
            print(f'  SKIP: bridge angle calculation failed: {ba["warnings"]}')

    # Test 7: Space group check
    print('\n--- Test 7: Space Group Check (spglib) ---')
    if os.path.exists(test_contcar):
        struct = read_contcar(test_contcar)
        sg = check_space_group(struct, symprec_list=[1e-4, 1e-3, 1e-2])
        test('spglib version available', sg.get('spglib_version') is not None)
        if 'error' not in sg.get('nominal', {}):
            print(f'  spglib version: {sg["spglib_version"]}')
            print(f'  Nominal (1e-3): {sg["nominal"].get("international")} (#{sg["nominal"].get("number")})')
            print(f'  P2_1 stable range: {sg["p21_stable_range"]}')
        else:
            print(f'  spglib error: {sg["nominal"].get("error")}')

    # Test 8: Full system processing
    print('\n--- Test 8: Full System Processing ---')
    if os.path.exists(test_contcar):
        result = process_system(test_contcar, system_name='S_test')
        test('system name correct', result['system'] == 'S_test')
        test('n_pb == 2', result['n_pb'] == 2, f'got {result["n_pb"]}')
        test('bldi_mean not None', result['bldi_mean'] is not None)
        test('delta_inplane_mean not None', result['delta_inplane_mean'] is not None)
        test('contcar_sha256 present', len(result['contcar_sha256']) == 64)
        print(f'  BLDI mean = {result["bldi_mean"]:.6f}')
        print(f'  delta_inplane mean = {result["delta_inplane_mean"]:.6f} A')
        print(f'  delta_total mean = {result["delta_total_mean"]:.6f} A')
        bb = result["bridge_bending_mean"]
        bb_str = f'{bb:.4f} deg' if bb is not None else 'N/A'
        print(f'  bridge bending mean = {bb_str}')

    print('\n' + '=' * 60)
    print(f'RESULTS: {passed} passed, {failed} failed')
    print('=' * 60)
    return failed == 0


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified Structural Descriptor Calculator for 2D Hybrid Perovskites'
    )
    parser.add_argument('--contcar', type=str, help='Path to a single CONTCAR file')
    parser.add_argument('--batch', type=str,
                        help='Base directory containing opt_* subdirectories')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--spacegroup', action='store_true',
                        help='Also run space group check (slower)')

    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.contcar:
        result = process_system(args.contcar)
        print(f'\nSystem: {result["system"]}')
        print(f'  Volume: {result["volume"]:.2f} A^3')
        print(f'  BLDI (mean): {result["bldi_mean"]:.6f}')
        print(f'  delta_inplane (mean): {result["delta_inplane_mean"]:.6f} A')
        print(f'  delta_outofplane (mean): {result["delta_outofplane_mean"]:.6f} A')
        print(f'  delta_total (mean): {result["delta_total_mean"]:.6f} A')
        bb = result["bridge_bending_mean"]
        ba = result["bridge_anisotropy_mean"]
        bb_str = f'{bb:.4f} deg' if bb is not None else 'N/A'
        ba_str = f'{ba:.4f} deg' if ba is not None else 'N/A'
        print(f'  bridge_bending (mean): {bb_str}')
        print(f'  bridge_anisotropy (mean): {ba_str}')
        if result['warnings']:
            print(f'  Warnings ({len(result["warnings"])}):')
            for w in result['warnings'][:5]:
                print(f'    - {w}')
        if args.output:
            write_results_csv([result], args.output)
            print(f'\nResults written to {args.output}')

    elif args.batch:
        output = args.output or 'unified_descriptors_all9.csv'
        results = process_batch(args.batch, output_csv=output)
        print(f'\nProcessed {len(results)} systems')
    else:
        parser.print_help()
        print('\nExample:')
        print('  python unified_descriptors.py --test')
        print('  python unified_descriptors.py --contcar ../03_vasp_current/opt/opt_S/CONTCAR')
        print('  python unified_descriptors.py --batch ../03_vasp_current/opt --output all9.csv')


if __name__ == '__main__':
    main()
