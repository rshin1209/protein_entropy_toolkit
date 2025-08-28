# protein_entropy_toolkit

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Issues](https://img.shields.io/github/issues-raw/USER/traj2bat-entropy.svg)](../../issues)

End-to-end toolkit to:
- convert MD trajectories to **internal coordinates (BAT)**,
- enumerate **bonds / angles / torsions** from topology,
- compute **1D entropies** and **pairwise mutual information (MI)** with **Jacobian** + **Miller–Madow** corrections,
- estimate final configurational entropy via a **MIST**-style reduction.

**Inputs:** AMBER `*.nc` + one `*.prmtop`  
**Outputs:** `topology.txt`, `res_data.json`, `dof*.npy`, `em.npy`, and final entropy.

---

## Features
- Traj → **BAT**: bonds (Å), angles/dihedrals (radians)
- Topology enumeration: unique simple paths of length 2/3/4
- Residue/atom map (`res_data.json`) with **0-based** `atom_index`
- Optional filtering: **backbone** (N, CA, C, O) or **noH** (exclude H*)
- Entropy/MI (Numba): **Jacobian** (bond \(r^2\), angle \(\sin\theta\), torsion 1) + **Miller–Madow**
- Final entropy (kcal/mol) using **MIST** (max off-diagonal MI per DoF, diagonal excluded)

---

## Install

```bash
# (recommended) create env
conda create -n eps python=3.10 -y
conda activate eps

# core deps
pip install mdtraj numpy numba tqdm

# (conda-forge alternative)
# conda install -c conda-forge mdtraj numpy numba tqdm

## Quick-Start

# 1) Trajectory → BAT (writes to output/all/<SYSTEM>)
python scripts/traj2bat.py --system_name <SYSTEM> -v

# 2) (Optional) Filter to backbone or noH (writes to output/<MODE>/<SYSTEM>)
python scripts/filter_dofs.py <SYSTEM> --filter backbone -v
# or
python scripts/filter_dofs.py <SYSTEM> --filter noH -v

# 3) Entropy + Mutual Information (writes em.npy in chosen region)
python scripts/entropy_mi.py --system_name <SYSTEM> --region backbone --bins 50 -v

# 4) Final entropy (kcal/mol) with optional residue selection (1-based)
python scripts/final_entropy.py --system_name <SYSTEM> --region backbone --temperature 298.15 -v
# e.g. limit to residues 1–120 and 150:
python scripts/final_entropy.py --system_name <SYSTEM> --region backbone --temperature 298.15 --residues "1-120,150" -v
