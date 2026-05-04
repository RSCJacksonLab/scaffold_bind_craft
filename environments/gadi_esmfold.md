# Gadi ESMFold Environment

This setup avoids Docker and installs ESMFold into one Python environment used by
`scaffold_bind_craft/scripts/run_esmfold.py`.

Upstream ESM notes used for this environment:

- ESMFold is loaded as `esm.pretrained.esmfold_v1()`.
- ESMFold installs from `fair-esm[esmfold]`.
- The upstream repo says to start from Python `<=3.9` with PyTorch installed.
- OpenFold installation requires `nvcc` and a CUDA-compatible PyTorch install.
- Multimer sequences are represented as one FASTA sequence with chains separated
  by `:`.
- `model.set_chunk_size(128)` is the documented memory-saving option; lower
  values reduce memory further at the cost of speed.
- `esmfold_v1()` downloads `esmfold_3B_v1.pt` from
  `https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt` through
  `torch.hub` when the checkpoint is not already present.

References:

- https://github.com/facebookresearch/esm
- https://github.com/facebookresearch/esm#esmfold-structure-prediction
- https://opus.nci.org.au/spaces/Help/pages/248840704/Python

## Step 1: Create The Environment

Pick an install location on `/scratch` or another project filesystem, not
`$HOME`.

```bash
cd /path/to/install/location

module purge
module use -a /g/data/if89/apps/modulefiles

# ESMFold's upstream install notes require Python <= 3.9.
module load python3/3.9.2 cuda/12.9.0 gcc/15.1.0

which python3
python3 --version
which nvcc
nvcc --version

python3 -m venv esmfold
source esmfold/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first. Use the wheel/channel that matches the CUDA module and
the current NCI recommendation for Gadi. Confirm CUDA is visible before
installing ESMFold dependencies.

```bash
# Example only: choose the current CUDA-compatible PyTorch command for Gadi.
python -m pip install torch

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
PY
```

Install ESMFold and the OpenFold dependencies following the upstream ESM README.
Install OpenFold with build isolation disabled; its setup imports `torch`, and
pip's isolated build environment cannot see the Torch package already installed
in the `esmfold` venv.

```bash
python -m pip install "fair-esm[esmfold]"
python -m pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
python -m pip install --no-build-isolation 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

If OpenFold fails with `ModuleNotFoundError: No module named 'torch'` while
`pip list` shows Torch is installed, rerun only the OpenFold command with
`--no-build-isolation`.

## Step 2: Download Weights On copyq

Gadi compute queues generally do not have external network access. Download the
ESMFold checkpoint on `copyq` before running GPU jobs:

```bash
cd /path/to/repos/scaffold_bind_craft
qsub -v TORCH_HOME=/scratch/xz2/$USER/torch_cache environments/00_download_esmfold_weights_copyq.pbs
```

This writes:

```text
/scratch/xz2/$USER/torch_cache/hub/checkpoints/esmfold_3B_v1.pt
```

That path matches `torch.hub` lookup when `TORCH_HOME=/scratch/xz2/$USER/torch_cache`.

The first validation run below also checks that ESMFold can read the downloaded
checkpoint. It should not need network if `TORCH_HOME` points at the cache filled
by the copyq job.

```bash
export TORCH_HOME=/scratch/xz2/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/xz2/$USER/xdg_cache

python - <<'PY'
import esm
import torch

model = esm.pretrained.esmfold_v1()
model = model.eval()
model.set_chunk_size(128)
if torch.cuda.is_available():
    model = model.cuda()

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
with torch.no_grad():
    pdb = model.infer_pdb(sequence)

print(pdb.splitlines()[0])
print("ok")
PY
```

Keep caches on project storage if `$HOME` quota or inode pressure is a concern:

```bash
mkdir -p /scratch/xz2/$USER/torch_cache /scratch/xz2/$USER/xdg_cache
export TORCH_HOME=/scratch/xz2/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/xz2/$USER/xdg_cache
```

## Step 3: Update Settings

Use the `esmfold` block in the run settings:

```json
"esmfold": {
  "gpu_device": 0,
  "target_chains": "A",
  "target_sequences": null,
  "binder_chain": "B",
  "chunk_size": 128,
  "num_recycles": null,
  "max_sequences_per_backbone": null
}
```

Notes:

- `chunk_size` maps to ESMFold's `model.set_chunk_size(...)`.
- `num_recycles: null` leaves ESMFold at its default recycle count.
- `run_esmfold.py` writes multimer input FASTA records with chains separated by
  `:`, matching the ESMFold convention.
- ESMFold pLDDT is stored in the PDB B-factor column and is summarized into
  `esmfold_plddt`, `esmfold_binder_plddt`, and `esmfold_target_plddt`.

## Step 4: Clean The Environment

```bash
cd /path/to/install/location

chgrp -R xz2 ./esmfold
chmod -R g+rwX ./esmfold
find ./esmfold -type d -exec chmod g+s {} +

# Check inode usage before cleaning.
find ./esmfold -xdev -printf '.' | wc -c

find ./esmfold -type d -name "doc*" -exec rm -rf {} +
find ./esmfold -type d -name "tests" -exec rm -rf {} +
find ./esmfold -type d -name "__pycache__" -exec rm -rf {} +
find ./esmfold -name "*.pyc" -delete
find ./esmfold -name "*.pyo" -delete
find ./esmfold -name ".DS_Store" -delete
find ./esmfold -type d -name ".pytest_cache" -exec rm -rf {} +

# Check inode usage after cleaning.
find ./esmfold -xdev -printf '.' | wc -c
```

## Step 5: Load For A Run

```bash
module purge
module use -a /g/data/if89/apps/modulefiles
module load python3/3.9.2 cuda/12.9.0 gcc/15.1.0

export TORCH_HOME=/scratch/xz2/$USER/torch_cache
export XDG_CACHE_HOME=/scratch/xz2/$USER/xdg_cache
export TMPDIR="${PBS_JOBFS:-/tmp}"

source /path/to/install/location/esmfold/bin/activate

cd /path/to/repos/scaffold_bind_craft
python3 scripts/run_esmfold.py --settings settings/my_run.json --limit 100
```

For PBS submission, use `scripts/gadi/03_run_esmfold.pbs` and set `VENV` to the
`esmfold` environment path:

```bash
qsub -v VENV=/path/to/install/location/esmfold,SETTINGS=/path/to/settings.json,LIMIT=100 scripts/gadi/03_run_esmfold.pbs
```
