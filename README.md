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
```
## Quick-Start

```bash
# 1) Trajectory → BAT (writes to output/all/<SYSTEM>)
python xyz2bat.py --system_name <SYSTEM> -v

# 2) (Optional) Filter to backbone or noH (writes to output/<MODE>/<SYSTEM>)
python extract_region.py <SYSTEM> --filter backbone -v
# or
python extract_region.py <SYSTEM> --filter noH -v

# 3) Entropy + Mutual Information (writes em.npy in chosen region)
python entropy_sampler.py --system_name <SYSTEM> --region backbone --bins 50 -v

# 4) Final entropy (kcal/mol) with optional residue selection (1-based)
python entropy_compiler.py --system_name <SYSTEM> --region backbone --temperature 298.15 -v
# e.g. limit to residues 1–120 and 150:
python entropy_compiler.py --system_name <SYSTEM> --region backbone --temperature 298.15 --residues "1-120,150" -v

```

## Example Run (Model System: FC_WT/no exclusion: all)

```bash
# 1) Amber prmtop and nc clean up
parm fc_wt.prmtop
trajin md.nc 1 100000 1
strip !(:1-17)
trajout cleaned_nc netcdf4
run
quit

parm fc_wt.prmtop
parmstrip !(:1-17)
parmwrite out cleaned_prmtop.prmtop
run
quit

# 1) Trajectory → BAT (writes to output/all/fc_wt)
python traj2bat.py --system_name fc_wt -v

[INFO] Input directory:  dataset/fc_wt
[INFO] Output directory: output/all/fc_wt
[INFO] Using topology file: test.prmtop
[INFO] [1/1] Loading: test.nc
[INFO] Saved: output/all/fc_wt/topology.txt
[INFO] Saved: output/all/fc_wt/res_data.json
[INFO] Saved: output/all/fc_wt/dof1.npy (shape=(50000, 1385))
[INFO] All files processed successfully.

# 2) Entropy + Mutual Information (writes em.npy in chosen region)
python entropy_sampler.py --system_name fc_wt --bins 50 -v

[INFO] Data directory: output/all/fc_wt
[INFO] Loaded topology with 1385 DOFs.
[INFO] Computed jtype_list with 1385 entries.
[INFO] Loaded dof1.npy (shape=(500, 1385))
[INFO] DOFs stacked -> (1385, 500)
[INFO] Computing 1D entropies for 1385 DOFs with 50 bins.
H1D: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1385/1385 [00:01<00:00, 1333.20it/s]
[INFO] Computing pairwise MI (symmetric)...
MI: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1385/1385 [01:45<00:00, 13.16it/s]
[INFO] Saved entropy/MI matrix: output/all/fc_wt/em.npy (shape=(1385, 1385))

# 4) Final entropy (kcal/mol)
python entropy_sampler.py --system_name fc_wt --temperature 300.0 -v
[INFO] Temperature (K) = 300.00
[INFO] Original atom count (from topology): 250
[INFO] Loaded 1385 DOFs; em.npy shape: (1385, 1385)
[INFO] No residue selection: using all residues.
[INFO] Allowed atoms after residue selection: 250
[INFO] Effective atom_num for scaling = 250
[INFO] Original DOFs: 1385; Kept DOFs: 1385
[INFO] S1D_bond=-1550.293516, S1D_angle=-363.367220, S1D_torsion=11.532246
[INFO] MIST_bond=236.893183, MIST_angle=236.400570, MIST_torsion=310.480063
[INFO] FINAL ENTROPY (kcal/mol) = -1601.230950
```
