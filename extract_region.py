#!/usr/bin/env python3
"""
Filter topology/DoFs and write a filtered res_data.json.

Modes:
  - backbone: keep only atoms named {N, CA, C, O}
  - noH    : keep all atoms except those whose names start with 'H'/'h'

Input (required):
  ./output/all/{system_name}/topology.txt
  ./output/all/{system_name}/res_data.json
  ./output/all/{system_name}/*.npy (optional)

Output:
  ./output/{filter}/{system_name}/topology.txt
  ./output/{filter}/{system_name}/res_data.json
  ./output/{filter}/{system_name}/*_{filter}.npy
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


# ---------------------------- logging --------------------------------- #
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ------------------------- CLI / arguments ---------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract filtered topology and DoFs; create a filtered res_data.json."
    )
    p.add_argument(
        "system_name",
        type=str,
        help="Name of the system (subfolder in ./output/all).",
    )
    p.add_argument(
        "--filter",
        choices=["backbone", "noH"],
        default="backbone",
        help="Filtering mode: 'backbone' (N, CA, C, O) or 'noH' (exclude atoms starting with H).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v=INFO, -vv=DEBUG; default=INFO).",
    )
    return p.parse_args()


# --------------------------- I/O helpers ------------------------------ #
def in_dir(system_name: str) -> Path:
    return Path("./output/all") / system_name


def out_dir(system_name: str, mode: str) -> Path:
    return Path("./output") / mode / system_name


def read_json(path: Path) -> object:
    with path.open("r") as f:
        return json.load(f)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved: {path}")


def write_topology(lines: Sequence[str], atom_count: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{atom_count}\n")
        for ln in lines:
            f.write(ln + "\n")
    logging.info(f"Saved: {path}")


# --------------------------- core filters ----------------------------- #
def backbone_atom_filter(name: str) -> bool:
    return name in {"N", "CA", "C", "O"}


def noH_atom_filter(name: str) -> bool:
    return not name.upper().startswith("H")


def collect_selected_indices(
    residues_json: List[Dict[str, object]],
    mode: str,
) -> Tuple[Set[int], Set[int]]:
    """
    Returns:
      (selected_0based, selected_1based)
    """
    if mode == "backbone":
        accept = backbone_atom_filter
    else:
        accept = noH_atom_filter

    sel0: Set[int] = set()
    sel1: Set[int] = set()
    for res in residues_json:
        for atom in res.get("atoms", []):
            name = atom["atom_name"]
            idx0 = int(atom["atom_index"])
            if accept(name):
                sel0.add(idx0)
                sel1.add(idx0 + 1)
    logging.debug(f"Selected {len(sel1)} atoms (1-based) with mode='{mode}'")
    return sel0, sel1


def read_topology_lines(path: Path) -> Tuple[int, List[str]]:
    """
    Reads topology.txt:
      first line: total atom count (string/integer)
      remaining: space-separated 1-based atom indices (length 2/3/4 per line)
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty topology: {path}")
    try:
        total_atoms = int(lines[0].strip())
    except Exception:
        total_atoms = -1  # keep original as unknown; it’s not used for filtering
    dof_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
    return total_atoms, dof_lines


def filter_topology(
    dof_lines: Sequence[str],
    selected_1based: Set[int],
) -> Tuple[List[str], int]:
    """
    Keep only lines whose *all* 1-based atom indices are in `selected_1based`.
    Returns:
      (filtered_lines, filtered_atom_count)
    """
    filtered: List[str] = []
    used_atoms: Set[int] = set()

    for ln in dof_lines:
        toks = tuple(map(int, ln.split()))
        if all(a in selected_1based for a in toks):
            filtered.append(ln)
            used_atoms.update(toks)

    return filtered, len(used_atoms)


def build_line_index_map(dof_lines: Sequence[str]) -> Dict[Tuple[int, ...], int]:
    """
    Map from topology line tokens (tuple of ints) to original column index.
    Assumes original topology lines are unique.
    """
    m: Dict[Tuple[int, ...], int] = {}
    for i, ln in enumerate(dof_lines):
        toks = tuple(map(int, ln.split()))
        m[toks] = i
    return m


def extract_filtered_columns(
    npy_in_dir: Path,
    dof_lines: Sequence[str],
    filtered_lines: Sequence[str],
    out_dir_path: Path,
    mode: str,
) -> None:
    """
    For each *.npy in npy_in_dir:
      - validate (2D, num_dofs matches topology length)
      - slice columns corresponding to filtered_lines
      - save as *_<mode>.npy in out_dir_path
    """
    files = sorted(npy_in_dir.glob("*.npy"))
    if not files:
        logging.info(f"No .npy files found in {npy_in_dir}. Skipping DoF extraction.")
        return

    line_map = build_line_index_map(dof_lines)
    sel_cols: List[int] = []
    for ln in filtered_lines:
        t = tuple(map(int, ln.split()))
        if t not in line_map:
            logging.warning(f"Filtered line not in original topology: {ln}")
            continue
        sel_cols.append(line_map[t])

    if not sel_cols:
        logging.warning("No filtered DoF columns selected; skipping .npy extraction.")
        return

    out_dir_path.mkdir(parents=True, exist_ok=True)

    for npy_path in files:
        try:
            arr = np.load(npy_path, mmap_mode="r")
        except Exception as e:
            logging.error(f"Failed to load {npy_path.name}: {e}")
            continue

        if arr.ndim != 2:
            logging.warning(f"Skipping {npy_path.name}: not 2D (shape={arr.shape}).")
            continue

        n_frames, n_dofs = arr.shape
        if n_dofs != len(dof_lines):
            logging.warning(
                f"Skipping {npy_path.name}: DoF count mismatch ({n_dofs} != {len(dof_lines)})."
            )
            continue

        filtered_arr = np.asarray(arr[:, sel_cols])  # materialize to regular ndarray
        out_name = f"{npy_path.stem}_{mode}.npy"
        out_path = out_dir_path / out_name
        np.save(out_path, filtered_arr)
        logging.info(f"Saved: {out_path} (shape={filtered_arr.shape})")


def write_filtered_res_data(
    residues_json: List[Dict[str, object]],
    selected_0based: Set[int],
    out_path: Path,
) -> None:
    """
    Keep original 0-based atom_index values, but remove atoms not in selected_0based.
    Drop residues that end up empty.
    """
    filtered_residues: List[Dict[str, object]] = []
    for res in residues_json:
        atoms = [
            atom for atom in res.get("atoms", [])
            if int(atom["atom_index"]) in selected_0based
        ]
        if atoms:
            filtered_residues.append(
                {
                    "residue_name": res["residue_name"],
                    "residue_number": res["residue_number"],
                    "atoms": atoms,
                }
            )
    write_json(filtered_residues, out_path)


# ------------------------------- runner ------------------------------- #
def run(system_name: str, mode: str) -> None:
    src = in_dir(system_name)
    dst = out_dir(system_name, mode)

    # Inputs
    res_json_path = src / "res_data.json"
    topo_path = src / "topology.txt"

    if not res_json_path.is_file():
        raise FileNotFoundError(f"Missing: {res_json_path}")
    if not topo_path.is_file():
        raise FileNotFoundError(f"Missing: {topo_path}")

    # Load inputs
    residues = read_json(res_json_path)  # list[residue dicts]
    orig_atom_count, dof_lines = read_topology_lines(topo_path)

    # Select atoms
    sel0, sel1 = collect_selected_indices(residues, mode)

    # Filter topology lines
    filtered_lines, filtered_atom_count = filter_topology(dof_lines, sel1)

    # Write filtered topology
    write_topology(filtered_lines, filtered_atom_count, dst / "topology.txt")

    # Report
    logging.info(f"Filter mode            : {mode}")
    logging.info(f"Original total atoms   : {orig_atom_count}")
    logging.info(f"Original DOF lines     : {len(dof_lines)}")
    logging.info(f"Filtered total atoms   : {filtered_atom_count}")
    logging.info(f"Filtered DOF lines     : {len(filtered_lines)}")

    # Extract DoFs from .npy files
    extract_filtered_columns(src, dof_lines, filtered_lines, dst, mode)

    # Write filtered res_data.json
    write_filtered_res_data(residues, sel0, dst / "res_data.json")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(system_name=args.system_name, mode=args.filter)


if __name__ == "__main__":
    main()
