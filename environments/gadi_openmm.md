# Step 0:
Make sure it is not already installed!!! **It may already be partly available on Gadi**

This setup avoids Docker and Conda where possible. It uses Gadi modules for Python and CUDA, then uses a Python venv for OpenMM, PDBFixer, PyRosetta-free scoring dependencies, and `scaffold_bind_craft`.


# Step 1: Create env

```
# navigate to correct dir
cd /path/to/install/location

# load modules
module load python3/3.11.7 cuda/12.9.0

# check module binaries
which python3
python3 --version
nvcc --version

# make OpenMM env
python3 -m venv openmm_screen

# load env
source openmm_screen/bin/activate

# install dependencies
python -m pip install --upgrade pip setuptools wheel

# OpenMM screening dependencies
python -m pip install \
  "numpy<2.0.0" \
  scipy \
  pandas \
  biopython \
  openmm \
  freesasa

# PDBFixer is normally distributed from the OpenMM GitHub repository
python -m pip install "pdbfixer @ git+https://github.com/openmm/pdbfixer.git"

# leave the env
deactivate
```

If the GitHub PDBFixer install is blocked by login-node network policy, install PDBFixer on a node/session where outbound HTTPS is available, or use a prebuilt/shared wheel if one exists for the project.

Update the `tools` section of your settings JSON:

```
"tools": {
  "colabdesign_path": "/path/to/colabdesign_modified",
  "alphafold_params_dir": "/path/to/alphafold/params",
  "proteinmpnn_path": "/path/to/ProteinMPNN"
}
```

The OpenMM path can optionally use `FASPR` for side-chain repacking and `sc` / `sc-rs` for shape complementarity if those binaries are available on `PATH`, or via `FASPR_BIN` and `SC_RS_BIN`.

```
export FASPR_BIN=/path/to/FASPR
export SC_RS_BIN=/path/to/sc
```

Check that the environment imports the required packages and sees a GPU-capable OpenMM platform:

```
module load python3/3.11.7 cuda/12.9.0
source /path/to/install/location/openmm_screen/bin/activate

python - <<'PY'
import openmm
from openmm import Platform
from pdbfixer import PDBFixer
import Bio
import scipy
import pandas

print("OpenMM", openmm.version.version)
print("Platforms", [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
PY

deactivate
```

The platform list should include `CUDA` on a GPU node. On a login node it may only show CPU/reference platforms, depending on Gadi's driver visibility.


# Step 2: Clean env
```
# change group to xz2 and give read write privileges
chgrp -R xz2 ./openmm_screen
chmod -R g+rwX ./openmm_screen
find ./openmm_screen -type d -exec chmod g+s {} +

# check inode usage before cleaning
find ./openmm_screen -xdev -printf '.' | wc -c

# Remove documentation directories
find ./openmm_screen -type d -name "doc*" -exec rm -rf {} +

# Remove test directories
find ./openmm_screen -type d -name "tests" -exec rm -rf {} +

# Remove Python cache files
find ./openmm_screen -type d -name "__pycache__" -exec rm -rf {} +

# Remove additional cache and temp files
find ./openmm_screen -name "*.pyc" -delete
find ./openmm_screen -name "*.pyo" -delete
find ./openmm_screen -name ".DS_Store" -delete
find ./openmm_screen -type d -name ".pytest_cache" -exec rm -rf {} +

# check inode usage after cleaning
find ./openmm_screen -xdev -printf '.' | wc -c
```


# Step 3: Loading Dir
```
module load python3/3.11.7 cuda/12.9.0
source /path/to/install/location/openmm_screen/bin/activate

# use node-local temporary space inside PBS jobs
export TMPDIR="${PBS_JOBFS:-/tmp}"
```

For the existing PBS script, set `VENV` to this OpenMM environment:

```
qsub -v VENV=/path/to/install/location/openmm_screen,SETTINGS=/path/to/settings.json,LIMIT=100 scripts/gadi/04_run_openmm_screen.pbs
```

If you want OpenMM screening outputs written somewhere other than the settings file output directory, pass `OUTPUT_DIR`:

```
qsub -v VENV=/path/to/install/location/openmm_screen,SETTINGS=/path/to/settings.json,OUTPUT_DIR=/scratch/xz2/$USER/runs/run_001,LIMIT=100 scripts/gadi/04_run_openmm_screen.pbs
```
