#!/usr/bin/env python3
"""
Entropy & Mutual Information over internal coordinates (DoFs).

Reads:
  ./output/{region}/{system_name}/topology.txt
  ./output/{region}/{system_name}/dof*.npy

Writes:
  ./output/{region}/{system_name}/em.npy     # entropy/MI matrix

Notes
-----
- 1D entropies with Jacobian correction (bond r^2, angle sin(theta), torsion 1)
- 2D joint entropies similarly corrected
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import trange
from numba import njit, prange

# ------------------------------- Consts -------------------------------- #

DOF_TYPE_MAPPING = {"bond": 0, "angle": 1, "torsion": 2}
REGION_DIR = {"all": Path("./output/all"), "backbone": Path("./output/backbone"), "noH": Path("./output/noH")}

# ------------------------------ Logging -------------------------------- #

def setup_logging(verbosity: int = 1) -> None:
    """
    -v  => INFO
    -vv => DEBUG
    (default INFO)
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

# ------------------------------- CLI ----------------------------------- #

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perform entropy/MI calculation over DoFs.")
    p.add_argument("--system_name", type=str, required=True, help="Name of the system.")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory containing topology.txt and dof*.npy. Overrides --region.",
    )
    p.add_argument(
        "--region",
        choices=["all", "backbone", "noH"],
        default="all",
        help="Base path is ./output/{region}/{system_name} when --output_dir is not provided.",
    )
    p.add_argument("--bins", type=int, default=50, help="Histogram bins for entropy/MI.")
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
    )
    return p.parse_args()

# --------------------------- DOF type utils ---------------------------- #

def determine_jtype(dof_tokens: List[str]) -> int:
    """
    Decide DOF type from length of tokens: 2=bond, 3=angle, 4=torsion.
    Falls back to 'bond' when unknown.
    """
    n = len(dof_tokens)
    jtype_str = {2: "bond", 3: "angle", 4: "torsion"}.get(n, "bond")
    if jtype_str == "bond" and n not in (2, 3, 4):
        logging.warning(f"Unknown DOF type for {dof_tokens}; defaulting to 'bond'.")
    return DOF_TYPE_MAPPING[jtype_str]

# ----------------------------- Numba kernels --------------------------- #
# Use math from numba (np.* where supported). Avoid Python closures.

@njit
def _sin(x: float) -> float:
    return np.sin(x)

@njit
def get_jacobian_1D(jtype: int, bin_index: int, min_val: float, dx: float) -> float:
    """
    Jacobian at the 1D histogram bin center.
      bond    -> r^2
      angle   -> sin(theta)
      torsion -> 1
    """
    if jtype == 0:  # bond
        x_center = min_val + dx * (bin_index + 0.5)
        return x_center * x_center
    elif jtype == 1:  # angle
        x_center = min_val + dx * (bin_index + 0.5)
        return _sin(x_center)
    else:  # torsion or unknown
        return 1.0

@njit
def get_jacobian_2D(i: int, j: int, xjtype: int, yjtype: int,
                    min_x: float, dx: float, min_y: float, dy: float) -> float:
    """
    2D Jacobian at (i,j) bin centers.
    """
    x_center = min_x + dx * (i + 0.5)
    y_center = min_y + dy * (j + 0.5)

    if xjtype == 0 and yjtype == 0:   # bond-bond
        return (x_center * x_center) * (y_center * y_center)
    elif xjtype == 0 and yjtype == 1: # bond-angle
        return (x_center * x_center) * _sin(y_center)
    elif xjtype == 0 and yjtype == 2: # bond-torsion
        return (x_center * x_center)
    elif xjtype == 1 and yjtype == 1: # angle-angle
        return _sin(x_center) * _sin(y_center)
    elif xjtype == 1 and yjtype == 2: # angle-torsion
        return _sin(x_center)
    elif xjtype == 2 and yjtype == 2: # torsion-torsion
        return 1.0
    else:
        return 1.0

@njit(parallel=True)
def compute_entropy_1D_numba(dof: np.ndarray, jtype: int, bins: int, min_val: float, dx: float) -> float:
    """
    1D entropy via histogram + Jacobian correction + Miller–Madow.
    """
    counts = np.zeros(bins, dtype=np.int64)
    n_samples = dof.shape[0]

    for idx in range(n_samples):
        x = dof[idx]
        bin_idx = int((x - min_val) / dx) if dx > 0 else 0
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= bins:
            bin_idx = bins - 1
        counts[bin_idx] += 1

    sample_size = 0
    for k in range(bins):
        sample_size += counts[k]
    if sample_size == 0:
        return 0.0

    entropy_sum = 0.0
    for i in prange(bins):
        c = counts[i]
        if c > 0:
            p = c / sample_size
            jacobian = get_jacobian_1D(jtype, i, min_val, dx if dx > 0 else 1.0)
            entropy_sum += -p * np.log(p / (jacobian * (dx if dx > 0 else 1.0)))

    # Correction
    n_non_zero = 0
    for i in range(bins):
        if counts[i] > 0:
            n_non_zero += 1
    entropy_sum += (n_non_zero - 1) / (2.0 * sample_size)
    return entropy_sum

@njit
def compute_joint_entropy(dof1: np.ndarray, dof2: np.ndarray,
                          xjtype: int, yjtype: int,
                          bins: int, min_x: float, dx: float, min_y: float, dy: float) -> float:
    """
    H(X,Y) via 2D histogram + Jacobian + Miller–Madow.
    """
    H_XY = np.zeros((bins, bins), dtype=np.float64)
    n_samples = dof1.shape[0]

    for idx in range(n_samples):
        x = dof1[idx]
        y = dof2[idx]
        bin_x = int((x - min_x) / dx) if dx > 0 else 0
        bin_y = int((y - min_y) / dy) if dy > 0 else 0
        if bin_x < 0:
            bin_x = 0
        elif bin_x >= bins:
            bin_x = bins - 1
        if bin_y < 0:
            bin_y = 0
        elif bin_y >= bins:
            bin_y = bins - 1
        H_XY[bin_x, bin_y] += 1.0

    total = 0.0
    for i in range(bins):
        for j in range(bins):
            total += H_XY[i, j]
    if total <= 0.0:
        return 0.0

    # convert to prob + entropy with Jacobian
    inv_total = 1.0 / total
    entropy_sum = 0.0
    for i in range(bins):
        for j in range(bins):
            p = H_XY[i, j] * inv_total
            if p > 0.0:
                jac = get_jacobian_2D(i, j, xjtype, yjtype, min_x, dx if dx > 0 else 1.0, min_y, dy if dy > 0 else 1.0)
                entropy_sum += -p * np.log(p / (jac * (dx if dx > 0 else 1.0) * (dy if dy > 0 else 1.0)))

    # Correction
    n_non_zero = 0
    for i in range(bins):
        for j in range(bins):
            if H_XY[i, j] > 0.0:
                n_non_zero += 1
    entropy_sum += (n_non_zero - 1) / (2.0 * total)
    return entropy_sum

@njit
def compute_mutual_information(dof1: np.ndarray, dof2: np.ndarray, H_X: float, H_Y: float,
                               xjtype: int, yjtype: int, bins: int,
                               min_x: float, dx: float, min_y: float, dy: float) -> float:
    H_XY = compute_joint_entropy(dof1, dof2, xjtype, yjtype, bins, min_x, dx, min_y, dy)
    return H_X + H_Y - H_XY

@njit
def precompute_min_max_dx(dofs: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-DOF min, max, and dx (bin width).
    """
    n_dofs = dofs.shape[0]
    min_vals = np.empty(n_dofs, dtype=np.float64)
    max_vals = np.empty(n_dofs, dtype=np.float64)
    dx_vals  = np.empty(n_dofs, dtype=np.float64)
    inv_bins = 1.0 / bins if bins > 0 else 1.0

    for i in range(n_dofs):
        mn = dofs[i, 0]
        mx = dofs[i, 0]
        for k in range(dofs.shape[1]):
            v = dofs[i, k]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        min_vals[i] = mn
        max_vals[i] = mx
        rng = mx - mn
        dx_vals[i] = rng * inv_bins if bins > 0 else 1.0
    return min_vals, max_vals, dx_vals

@njit(parallel=True)
def compute_mi_for_i(i: int,
                     dofs: np.ndarray,
                     jtype_list: np.ndarray,
                     entropy1d: np.ndarray,
                     bins: int,
                     min_vals: np.ndarray,
                     dx_vals: np.ndarray,
                     em: np.ndarray) -> None:
    """
    Compute MI for row i against all j>i (symmetric fill).
    """
    dof1 = dofs[i]
    jtype_i = jtype_list[i]
    H_X = entropy1d[i]
    min_x = min_vals[i]
    dx = dx_vals[i]
    n_dofs = dofs.shape[0]

    for j in prange(i + 1, n_dofs):
        dof2 = dofs[j]
        jtype_j = jtype_list[j]
        H_Y = entropy1d[j]
        min_y = min_vals[j]
        dy = dx_vals[j]
        mi = compute_mutual_information(dof1, dof2, H_X, H_Y, jtype_i, jtype_j, bins, min_x, dx, min_y, dy)
        em[i, j] = mi
        em[j, i] = mi

# ------------------------------- I/O ----------------------------------- #

def resolve_data_dir(system_name: str, region: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    base = REGION_DIR[region]
    return base / system_name

def load_topology(dir_path: Path) -> List[List[str]]:
    topo_file = dir_path / "topology.txt"
    if not topo_file.is_file():
        raise FileNotFoundError(f"Topology file not found: {topo_file}")
    lines = topo_file.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty topology file: {topo_file}")
    dof_list = [ln.strip().split() for ln in lines[1:] if ln.strip()]
    logging.info(f"Loaded topology with {len(dof_list)} DOFs.")
    return dof_list

def compute_jtype_list(dof_list: List[List[str]]) -> np.ndarray:
    jtypes = np.fromiter((determine_jtype(d) for d in dof_list), dtype=np.int32, count=len(dof_list))
    logging.info(f"Computed jtype_list with {jtypes.size} entries.")
    return jtypes

def load_dofs(dir_path: Path) -> np.ndarray:
    """
    Stack dof*.npy along frames and transpose to (num_dofs, total_frames).
    """
    files = sorted(dir_path.glob("dof*.npy"))
    if not files:
        raise FileNotFoundError(f"No DOF .npy files found in {dir_path}")
    data = []
    for f in files:
        arr = np.load(f, mmap_mode="r")
        if arr.ndim != 2:
            logging.warning(f"Skipping {f.name}: not 2D (shape={arr.shape}).")
            continue
        data.append(np.asarray(arr))  # materialize to regular ndarray
        logging.info(f"Loaded {f.name} (shape={arr.shape})")
    if not data:
        raise RuntimeError(f"No valid 2D DOF arrays in {dir_path}")
    dofs = np.vstack(data).T  # (sum_frames, dofs) -> (dofs, sum_frames)
    logging.info(f"DOFs stacked -> {dofs.shape}")
    return dofs

def save_entropy_matrix(em: np.ndarray, dir_path: Path) -> Path:
    out_file = dir_path / "em.npy"
    np.save(out_file, em)
    logging.info(f"Saved entropy/MI matrix: {out_file} (shape={em.shape})")
    return out_file

# --------------------------- High-level ops ---------------------------- #

def compute_entropy_1D_all(dofs: np.ndarray, jtypes: np.ndarray, bins: int) -> np.ndarray:
    """
    1D entropy per DOF (returns length = num_dofs).
    """
    num_dofs = dofs.shape[0]
    ent = np.empty(num_dofs, dtype=np.float64)
    logging.info(f"Computing 1D entropies for {num_dofs} DOFs with {bins} bins.")
    for i in trange(num_dofs, desc="H1D"):
        series = dofs[i]
        mn = np.min(series)
        mx = np.max(series)
        dx = (mx - mn) / bins if bins > 0 else 1.0
        ent[i] = compute_entropy_1D_numba(series.astype(np.float64), int(jtypes[i]), bins, float(mn), float(dx))
    return ent

def validate_dof_count_matches_topology(dofs: np.ndarray, dof_list: List[List[str]]) -> None:
    """
    Ensure number of topology lines equals number of DoF columns.
    """
    num_dofs_topo = len(dof_list)
    num_dofs_data = dofs.shape[0]
    if num_dofs_topo != num_dofs_data:
        logging.warning(f"#DoFs mismatch: topology={num_dofs_topo} vs data={num_dofs_data}")

# -------------------------------- Main -------------------------------- #

def main() -> None:
    args = parse_arguments()
    setup_logging(args.verbose)

    data_dir = resolve_data_dir(args.system_name, args.region, args.output_dir)
    if not data_dir.is_dir():
        logging.error(f"Data directory does not exist: {data_dir}")
        raise SystemExit(1)

    logging.info(f"Data directory: {data_dir}")

    # Load inputs
    dof_list = load_topology(data_dir)
    jtype_list = compute_jtype_list(dof_list)
    dofs = load_dofs(data_dir).astype(np.float64)

    validate_dof_count_matches_topology(dofs, dof_list)

    # 1) 1D entropies
    bins = int(args.bins)
    entropy1d = compute_entropy_1D_all(dofs, jtype_list, bins)

    # (Optional) pre-JIT warmups
    _ = compute_entropy_1D_numba(np.zeros(10, dtype=np.float64), 0, 10, 0.0, 1.0)
    _ = compute_mutual_information(
        np.zeros(10, dtype=np.float64), np.zeros(10, dtype=np.float64),
        1.0, 1.0, 0, 0, 10, 0.0, 1.0, 0.0, 1.0
    )

    # 2) Pairwise MI
    n_dofs = dofs.shape[0]
    em = np.zeros((n_dofs, n_dofs), dtype=np.float64)

    # place H(X) on diagonal
    for i in range(n_dofs):
        em[i, i] = entropy1d[i]

    # precompute per-DOF ranges
    min_vals, _max_vals, dx_vals = precompute_min_max_dx(dofs, bins)

    logging.info("Computing pairwise MI (symmetric)...")
    for i in trange(n_dofs, desc="MI"):
        compute_mi_for_i(i, dofs, jtype_list.astype(np.int32), entropy1d, bins, min_vals, dx_vals, em)

    # Save
    save_entropy_matrix(em, data_dir)

if __name__ == "__main__":
    main()
