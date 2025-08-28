#!/usr/bin/env python3
"""
Compute final configurational entropy from:
  - em.npy  (entropy/MI matrix; diagonal = H1D, off-diagonal used by MIST)
  - topology.txt (DOF definitions)
  - res_data.json (residues/atoms; 0-based atom_index)

Supports selecting residues (1-based, e.g. "1,2,10-20") and region paths:
  ./output/{all|backbone|noH}/{system_name}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

# Boltzmann constant in kcal/(mol*K)
kB = 1.987204259e-3

REGION_DIR = {"all": Path("./output/all"), "backbone": Path("./output/backbone"), "noH": Path("./output/noH")}


# ------------------------------- Logging -------------------------------- #

def setup_logging(verbosity: int = 1) -> None:
    """
    -v  => INFO
    -vv => DEBUG
    default INFO
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute final entropy from em.npy & topology.txt with optional residue selection."
    )
    p.add_argument("--system_name", type=str, required=True, help="System name (subfolder under region).")
    p.add_argument(
        "--region",
        choices=["all", "backbone", "noH"],
        default="all",
        help="Base path is ./output/{region}/{system_name}",
    )
    p.add_argument("--temperature", type=float, required=True, help="Temperature in Kelvin.")
    p.add_argument(
        "--residues",
        type=str,
        default=None,
        help="Residue indices (1-based), e.g. '1,2,10-20'. Omit to include all.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
    )
    return p.parse_args()


# ------------------------------- Utils --------------------------------- #

def parse_residue_range(spec: str | None) -> Set[int]:
    """
    '1,2,5-10' -> {1,2,5,6,...,10}; None/'' -> empty set (means 'no filtering')
    """
    if not spec:
        return set()
    out: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return out


def classify_dof(tokens: Sequence[str]) -> int:
    """
    Return 0=bond, 1=angle, 2=torsion based on token count.
    """
    n = len(tokens)
    if n == 2:
        return 0
    if n == 3:
        return 1
    if n == 4:
        return 2
    logging.warning(f"Unexpected DOF length {n} for tokens {tokens}; defaulting to bond (0).")
    return 0


def resolve_data_dir(system_name: str, region: str) -> Path:
    return REGION_DIR[region] / system_name


# ------------------------------- I/O ----------------------------------- #

def load_inputs(dir_path: Path) -> Tuple[List[str], np.ndarray, list]:
    """
    Load topology lines (without first line), em matrix, and residues json.
    """
    topo = dir_path / "topology.txt"
    em = dir_path / "em.npy"
    res = dir_path / "res_data.json"

    for p in (topo, em, res):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    topo_lines = [ln.strip() for ln in topo.read_text().splitlines() if ln.strip()]
    if not topo_lines:
        raise ValueError(f"Empty topology file: {topo}")

    # first line is atom count (may be used for logging only)
    try:
        atom_count_orig = int(topo_lines[0])
    except Exception:
        atom_count_orig = -1
        logging.warning("Failed to parse first line of topology as atom count.")

    dof_lines = topo_lines[1:]

    em_mat = np.load(em)
    if em_mat.ndim != 2 or em_mat.shape[0] != em_mat.shape[1]:
        raise ValueError(f"em.npy must be square 2D; got shape {em_mat.shape}")
    if em_mat.shape[0] != len(dof_lines):
        raise ValueError(f"em size {em_mat.shape[0]} != #DOF lines {len(dof_lines)}")

    residues = json.loads(res.read_text())

    logging.info(f"Original atom count (from topology): {atom_count_orig}")
    logging.info(f"Loaded {len(dof_lines)} DOFs; em.npy shape: {em_mat.shape}")
    logging.debug(f"Data directory: {dir_path}")

    return dof_lines, em_mat, residues


# ------------------------------ Core logic ----------------------------- #

def allowed_atoms_from_residues(residues: list, selection_1based: Set[int]) -> Set[int]:
    """
    Build a set of allowed 1-based atom indices from res_data.json given a residue selection.
    If selection is empty -> all residues are included.
    """
    allowed: Set[int] = set()
    for r in residues:
        rnum = int(r["residue_number"])  # 1-based in JSON
        if selection_1based and rnum not in selection_1based:
            continue
        for atom in r.get("atoms", []):
            idx0 = int(atom["atom_index"])      # 0-based
            allowed.add(idx0 + 1)               # convert to 1-based
    logging.info(f"Allowed atoms after residue selection: {len(allowed)}")
    return allowed


def filter_dofs(dof_lines: List[str], allowed_atoms_1b: Set[int]) -> Tuple[List[int], List[int]]:
    """
    Keep DOFs whose ALL atoms are in allowed set.
    Returns (kept_indices, kept_types)
    """
    kept_idx: List[int] = []
    kept_types: List[int] = []
    for i, line in enumerate(dof_lines):
        toks = line.split()
        atoms = tuple(map(int, toks))
        if all(a in allowed_atoms_1b for a in atoms):
            kept_idx.append(i)
            kept_types.append(classify_dof(toks))
    logging.info(f"Original DOFs: {len(dof_lines)}; Kept DOFs: {len(kept_idx)}")
    return kept_idx, kept_types


def refactor_sum(value: float, numerator: int, denominator: int, label: str) -> float:
    """
    Multiply `value` by numerator/denominator when both > 0; log otherwise.
    """
    if denominator > 0 and numerator > 0:
        return value * (numerator / denominator)
    logging.warning(f"Skipping {label} refactor => division by zero (num={numerator}, den={denominator}).")
    return value


def max_off_diagonal(row: np.ndarray, i: int) -> float:
    """
    Return max off-diagonal value in a symmetric row i (exclude j==i).
    """
    # Concatenate left and right parts excluding diagonal
    left_max = row[:i].max() if i > 0 else -np.inf
    right_max = row[i+1:].max() if i + 1 < row.shape[0] else -np.inf
    return max(left_max, right_max)


def compute_entropy_components(
    em_sub: np.ndarray,
    kept_types: List[int],
    atom_num_effective: int,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute S1D and MIST components per DOF type with refactoring.
    Returns:
      S1D_b, S1D_a, S1D_t, MIST_b, MIST_a, MIST_t
    """
    # Raw sums by type
    S1D_b = S1D_a = S1D_t = 0.0
    for i, t in enumerate(kept_types):
        eii = float(em_sub[i, i])
        if t == 0:
            S1D_b += eii
        elif t == 1:
            S1D_a += eii
        else:
            S1D_t += eii

    # Counts per type
    num_b = sum(1 for t in kept_types if t == 0)
    num_a = sum(1 for t in kept_types if t == 1)
    num_t = sum(1 for t in kept_types if t == 2)

    # Refactor (scaling by counts expected for an N-atom system)
    S1D_b = refactor_sum(S1D_b, atom_num_effective - 1, num_b, "S1D_bond")
    S1D_a = refactor_sum(S1D_a, atom_num_effective - 2, num_a, "S1D_angle")
    S1D_t = refactor_sum(S1D_t, atom_num_effective - 3, num_t, "S1D_torsion")

    # MIST: for each i, take max off-diagonal from row i
    MIST_b = MIST_a = MIST_t = 0.0
    for i, t in enumerate(kept_types):
        m = max_off_diagonal(em_sub[i], i)
        if t == 0:
            MIST_b += m
        elif t == 1:
            MIST_a += m
        else:
            MIST_t += m

    # Refactor MIST similarly
    MIST_b = refactor_sum(MIST_b, atom_num_effective - 1, num_b, "MIST_bond")
    MIST_a = refactor_sum(MIST_a, atom_num_effective - 2, num_a, "MIST_angle")
    MIST_t = refactor_sum(MIST_t, atom_num_effective - 3, num_t, "MIST_torsion")

    return S1D_b, S1D_a, S1D_t, MIST_b, MIST_a, MIST_t


# -------------------------------- Main -------------------------------- #

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data_dir = resolve_data_dir(args.system_name, args.region)
    if not data_dir.is_dir():
        logging.error(f"Data directory not found: {data_dir}")
        raise SystemExit(1)

    T = float(args.temperature)
    logging.info(f"Temperature (K) = {T:.2f}")

    # Load inputs
    dof_lines, em, residues = load_inputs(data_dir)

    # Residue selection (1-based numbers). Empty set -> all residues.
    selected_res = parse_residue_range(args.residues)
    if selected_res:
        logging.info(f"Residue selection: {sorted(selected_res)}")
    else:
        logging.info("No residue selection: using all residues.")

    # Allowed atoms (1-based) from selection
    allowed_atoms_1b = allowed_atoms_from_residues(residues, selected_res)
    atom_num_effective = len(allowed_atoms_1b)
    logging.info(f"Effective atom_num for scaling = {atom_num_effective}")

    # Filter DOFs to those fully within the allowed atom set
    kept_indices, kept_types = filter_dofs(dof_lines, allowed_atoms_1b)
    if not kept_indices:
        logging.error("No DOFs remain after filtering; cannot compute entropy.")
        raise SystemExit(1)

    idx = np.asarray(kept_indices, dtype=int)
    em_sub = em[np.ix_(idx, idx)]

    # Compute components
    S1D_b, S1D_a, S1D_t, MIST_b, MIST_a, MIST_t = compute_entropy_components(
        em_sub, kept_types, atom_num_effective
    )

    total_S1D = S1D_b + S1D_a + S1D_t
    total_MIST = MIST_b + MIST_a + MIST_t
    final_entropy = T * kB * (total_S1D - total_MIST)

    logging.info(f"S1D_bond={S1D_b:.6f}, S1D_angle={S1D_a:.6f}, S1D_torsion={S1D_t:.6f}")
    logging.info(f"MIST_bond={MIST_b:.6f}, MIST_angle={MIST_a:.6f}, MIST_torsion={MIST_t:.6f}")
    logging.info(f"FINAL ENTROPY (kcal/mol) = {final_entropy:.6f}")


if __name__ == "__main__":
    main()
