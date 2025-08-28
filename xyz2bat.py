#!/usr/bin/env python3
"""
Convert XYZ (MDTraj) trajectories to internal coordinates (BAT).

- Scans ./dataset/{system_name} for *.nc and exactly one *.prmtop
- Computes a connectivity-derived "topology" (pairs, triplets, quadruplets)
- Extracts residue/atom metadata -> res_data.json
- Saves DoFs (bonds[Å], angles[rad], dihedrals[rad]) per input file as dof{idx}.npy
- Writes topology.txt (1-based indices)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import mdtraj as md
import numpy as np


# ---------------------------- Logging --------------------------------- #
def setup_logging(verbosity: int = 1) -> None:
    """Configure root logger."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ---------------------------- I/O helpers ----------------------------- #
def find_input_files(input_dir: Path) -> Tuple[List[Path], Path]:
    """Find .nc and .prmtop in input_dir. Require >=1 nc and exactly one prmtop."""
    nc_files = sorted(input_dir.glob("*.nc"))
    top_files = sorted(input_dir.glob("*.prmtop"))

    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")
    if len(top_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .prmtop file in {input_dir}, found {len(top_files)}"
        )

    logging.info(f"Using topology file: {top_files[0].name}")
    return nc_files, top_files[0]


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved: {path}")


def save_topology_txt(atom_count: int, paths: Sequence[Tuple[int, ...]], path: Path) -> None:
    """Save 1-based topology lines after atom_count header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{atom_count}\n")
        for p in paths:
            f.write(" ".join(str(i + 1) for i in p) + "\n")
    logging.info(f"Saved: {path}")


# ---------------------- Topology & connectivity ----------------------- #
def generate_connectivity(traj: md.Trajectory) -> Dict[int, List[int]]:
    """Adjacency dict from MDTraj bonds."""
    graph: Dict[int, List[int]] = {}
    for bond in traj.top.bonds:
        i, j = bond.atom1.index, bond.atom2.index
        graph.setdefault(i, []).append(j)
        graph.setdefault(j, []).append(i)
    logging.debug("Connectivity graph generated.")
    return graph


def _dfs_paths_from(
    connectivity: Dict[int, List[int]], start: int, max_len: int, seen_paths: Set[Tuple[int, ...]]
) -> None:
    """Enumerate unique simple paths up to length max_len (inclusive) starting at node."""

    def dfs(path: List[int]) -> None:
        if len(path) > max_len:
            return
        if 2 <= len(path) <= max_len:
            t = tuple(path)
            rt = tuple(reversed(path))
            # store only one orientation
            if rt not in seen_paths:
                seen_paths.add(t)
        last = path[-1]
        for nbr in connectivity.get(last, []):
            if nbr not in path:
                path.append(nbr)
                dfs(path)
                path.pop()

    dfs([start])


def generate_topology(
    connectivity: Dict[int, List[int]], max_path_len: int = 4
) -> Tuple[int, List[Tuple[int, ...]]]:
    """
    Build unique (unordered) simple paths of lengths 2..max_path_len.
    Returns (atom_count, sorted_paths).
    """
    unique_paths: Set[Tuple[int, ...]] = set()
    for node in connectivity:
        _dfs_paths_from(connectivity, node, max_path_len, unique_paths)

    # sort first by length, then lexicographically
    sorted_paths = sorted(unique_paths, key=lambda x: (len(x), x))
    atom_count = len(connectivity)
    logging.debug(f"Generated topology with {atom_count} atoms and {len(sorted_paths)} paths.")
    return atom_count, sorted_paths


# ----------------------- Metadata extraction -------------------------- #
def extract_residue_data(traj: md.Trajectory) -> List[Dict[str, object]]:
    """Residue -> atoms metadata."""
    out: List[Dict[str, object]] = []
    top = traj.topology
    for res in top.residues:
        res_entry = {
            "residue_name": res.name,
            "residue_number": res.index + 1,  # 1-based
            "atoms": [{"atom_name": a.name, "atom_index": a.index} for a in res.atoms],
        }
        out.append(res_entry)
        logging.debug(f"Residue extracted: {res.name} #{res.index + 1}")
    return out


# ----------------------- BAT / internal coords ------------------------ #
def _split_paths_by_length(
    paths: Iterable[Tuple[int, ...]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    torsions: List[Tuple[int, int, int, int]] = []
    for p in paths:
        if len(p) == 2:
            bonds.append((p[0], p[1]))
        elif len(p) == 3:
            angles.append((p[0], p[1], p[2]))
        elif len(p) == 4:
            torsions.append((p[0], p[1], p[2], p[3]))
    return bonds, angles, torsions


def compute_and_save_dofs(traj: md.Trajectory, paths: Sequence[Tuple[int, ...]], out_path: Path) -> None:
    """Compute distances [Å], angles [rad], dihedrals [rad]; save npy."""
    bonds, angles, torsions = _split_paths_by_length(paths)

    arrs: List[np.ndarray] = []
    if bonds:
        d = md.compute_distances(traj, bonds) * 10.0  # nm -> Å
        arrs.append(d)
    if angles:
        a = md.compute_angles(traj, angles)
        arrs.append(a)
    if torsions:
        t = md.compute_dihedrals(traj, torsions)
        arrs.append(t)

    if not arrs:
        raise RuntimeError("No bond/angle/dihedral paths to compute.")

    dofs = np.hstack(arrs)
    dofs = dofs[~np.isnan(dofs).any(axis=1)]  # drop NaN rows if any
    np.save(out_path, dofs)
    logging.info(f"Saved: {out_path} (shape={dofs.shape})")


# ------------------------------ Runner -------------------------------- #
def run(system_name: str) -> None:
    input_dir = Path(f"./dataset/{system_name}")
    output_dir = Path(f"./output/all/{system_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input directory:  {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    nc_files, top_file = find_input_files(input_dir)

    topology_paths: List[Tuple[int, ...]] | None = None

    for idx, nc in enumerate(nc_files, start=1):
        logging.info(f"[{idx}/{len(nc_files)}] Loading: {nc.name}")
        try:
            traj = md.load(nc, top=top_file)
        except Exception as e:
            logging.error(f"Failed to load {nc}: {e}")
            continue

        # Build/save topology & residue metadata from the first trajectory only
        if topology_paths is None:
            conn = generate_connectivity(traj)
            atom_count, topology_paths = generate_topology(conn)
            save_topology_txt(atom_count, topology_paths, output_dir / "topology.txt")
            res_data = extract_residue_data(traj)
            save_json(res_data, output_dir / "res_data.json")

        # Save DoFs for this file
        out_npy = output_dir / f"dof{idx}.npy"
        compute_and_save_dofs(traj, topology_paths, out_npy)  # <-- fixed (use traj)

    logging.info("All files processed successfully.")


# ------------------------------- CLI ---------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert XYZ coordinates (MDTraj) to internal coordinates (BAT).")
    p.add_argument(
        "--system_name",
        type=str,
        required=True,
        help="System name (reads ./dataset/{system_name}, writes ./output/all/{system_name})",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v=INFO, -vv=DEBUG; default=INFO).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(system_name=args.system_name)


if __name__ == "__main__":
    main()
