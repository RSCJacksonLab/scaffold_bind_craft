# Step 0:
Make sure it is not already installed!!! **It may already be partly available on Gadi**

This setup avoids Docker. It uses Gadi modules for Python, CUDA, and MMseqs2, then uses pip for AlphaFast, Fast Protenix, and Protenix.

Use two Python environments:

- `fast_protenix_alphafast`: runs `scaffold_bind_craft`, `fast_protenix`, and AlphaFast preprocessing.
- `protenix`: runs the Protenix CLI for prediction.

Keeping them separate avoids dependency conflicts between AlphaFast and Protenix.


# Step 1: Create env

```
# navigate to correct dir
cd /path/to/install/location

# load modules
module use -a /g/data/if89/apps/modulefiles
module load python3/3.12.1 cuda/12.9.0 MMseqs2/18-8cc5c gcc/15.1.0

# check module binaries
which python3
which mmseqs
mmseqs version

# make AlphaFast / Fast Protenix env
python3 -m venv fast_protenix_alphafast

# load env
source fast_protenix_alphafast/bin/activate

# install dependencies
python -m pip install --upgrade pip setuptools wheel

# install AlphaFast and Fast Protenix from local checkouts
python -m pip install -e /path/to/repos/alphafast
python -m pip install -e /path/to/repos/fast_protenix

# leave the AlphaFast env
deactivate


# load new modules
module purge
module use -a /g/data/if89/apps/modulefiles
module load python3/3.10.4 cuda/12.9.0 MMseqs2/18-8cc5c gcc/15.1.0

# make and load env
python3 -m venv protenix
source protenix/bin/activate

# install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install protenix

# leave the Protenix env
deactivate
```

The AlphaFast database setup still needs the public AlphaFast databases and the padded MMseqs2 databases. If the prebuilt databases are being downloaded on Gadi, keep them on `/scratch` or another high-throughput project filesystem, not `$HOME`.

```
cd /path/to/repos/alphafast
module use -a /g/data/if89/apps/modulefiles
module load python3/3.12.1 cuda/12.9.0 MMseqs2/18-8cc5c
source /path/to/install/location/fast_protenix_alphafast/bin/activate

# default: protein + RNA MMseqs + mmCIF from HuggingFace
./scripts/setup_databases.sh /scratch/xz2/$USER/alphafast_databases

# expected protein MMseqs path for scaffold_bind_craft settings:
# /scratch/xz2/$USER/alphafast_databases/mmseqs
```

Update the `fast_protenix` section of your settings JSON:

```
"fast_protenix": {
  "path": "/path/to/repos/fast_protenix",
  "python": "/path/to/install/location/fast_protenix_alphafast/bin/python",
  "db_dir": "/scratch/xz2/YOUR_USER/alphafast_databases",
  "mmseqs_db_dir": "/scratch/xz2/YOUR_USER/alphafast_databases/mmseqs",
  "alphafast_dir": "/path/to/repos/alphafast",
  "batch_size": 512,
  "gpu_device": 0,
  "temp_dir": null,
  "run_protenix": true,
  "protenix_bin": "/path/to/install/location/protenix/bin/protenix",
  "protenix_model": "protenix_base_default_v1.0.0",
  "protenix_weights_dir": "/scratch/xz2/YOUR_USER/protenix_offline_data/checkpoint",
  "alphafast_args": [],
  "protenix_args": []
}
```

`protenix_weights_dir` should be the directory containing the Protenix checkpoint file, for example
`protenix_base_default_v1.0.0.pt`. It is passed to `protenix pred` as
`--load_checkpoint_dir`, so it can be used on airgapped machines instead of relying on
Protenix's default `~/checkpoint` location.


# Step 2: Clean env
```
# change group to xz2 and give read write privilages
chgrp -R xz2 ./fast_protenix_alphafast ./protenix
chmod -R g+rwX ./fast_protenix_alphafast ./protenix
find ./fast_protenix_alphafast ./protenix -type d -exec chmod g+s {} +

# check inode usage before cleaning
find ./fast_protenix_alphafast -xdev -printf '.' | wc -c
find ./protenix -xdev -printf '.' | wc -c

# Remove documentation directories
find ./fast_protenix_alphafast ./protenix -type d -name "doc*" -exec rm -rf {} +

# Remove test directories
find ./fast_protenix_alphafast ./protenix -type d -name "tests" -exec rm -rf {} +

# Remove Python cache files
find ./fast_protenix_alphafast ./protenix -type d -name "__pycache__" -exec rm -rf {} +

# Remove additional cache and temp files
find ./fast_protenix_alphafast ./protenix -name "*.pyc" -delete
find ./fast_protenix_alphafast ./protenix -name "*.pyo" -delete
find ./fast_protenix_alphafast ./protenix -name ".DS_Store" -delete
find ./fast_protenix_alphafast ./protenix -type d -name ".pytest_cache" -exec rm -rf {} +

# check inode usage after cleaning
find ./fast_protenix_alphafast -xdev -printf '.' | wc -c
find ./protenix -xdev -printf '.' | wc -c
```


# Step 3: Loading Dir
```
module use -a /g/data/if89/apps/modulefiles
module load python3/3.12.1 cuda/12.9.0 MMseqs2/18-8cc5c gcc/15.1.0

# run scaffold_bind_craft / Fast Protenix from the AlphaFast env
source /path/to/install/location/fast_protenix_alphafast/bin/activate

# load protenix
deactivate
module purge
module load python3/3.10.4 cuda/12.9.0 MMseqs2/18-8cc5c gcc/15.1.0
source protenix/bin/activate
```

# Step 4: Protenix Weights
Download to a different computer and move accross with rsync via gadi-dm
**Download**:
```
export PROTENIX_ROOT_DIR=~/protenix_models
mkdir -p "$PROTENIX_ROOT_DIR/checkpoint"

# Download Protenix v1 2025 checkpoint
curl -L -C - \
  -o "$PROTENIX_ROOT_DIR/checkpoint/protenix_base_20250630_v1.0.0.pt" \
  "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_20250630_v1.0.0.pt"

# Download inference accessory data for the 2025 model
curl -L -C - \
  -o "$PROTENIX_ROOT_DIR/common.tar.gz" \
  "https://protenix.tos-cn-beijing.volces.com/protenix.20260101/common.tar.gz"

curl -L -C - \
  -o "$PROTENIX_ROOT_DIR/search_database.tar.gz" \
  "https://protenix.tos-cn-beijing.volces.com/protenix.20260101/search_database.tar.gz"

# Extract accessory data
tar -xzvf "$PROTENIX_ROOT_DIR/common.tar.gz" -C "$PROTENIX_ROOT_DIR"
