# Gadi Chai-1 Offline Environment

This setup installs `git@github.com:RSCJacksonLab/chai-1-offline.git` for
`scripts/run_chai1_offline.py`. The stage runs in single-sequence mode with ESM
embeddings and does not call MSA or template servers during prediction.

## Requirements

- Linux
- Python 3.10 or later
- CUDA GPU with bfloat16 support
- `/scratch/xz2/chai-1_weights/` on shared storage for Chai assets and ESM-2 embedding weights

## Install

Run this on a Gadi node with internet access:

```bash
module load python3/3.10.4 cuda/12.9.0

cd /path/to/install/location
git clone git@github.com:RSCJacksonLab/chai-1-offline.git

python3 -m venv /path/to/install/location/chai1_offline
source /path/to/install/location/chai1_offline/bin/activate

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e /path/to/install/location/chai-1-offline
python -m pip install --no-cache-dir biopython
```

Download the Chai-1 assets and traced ESM-2 embedding model from `copyq`, which
has external network access. This does not run Chai inference:

```bash
qsub environments/00_download_chai1_weights_copyq.pbs
```

The downloader stages files with group `xz2` and group `rwx` permissions by
default. Override with `GROUP=...` if needed:

```bash
qsub -v WEIGHTS_DIR=/scratch/xz2/chai-1_weights/,GROUP=xz2 environments/00_download_chai1_weights_copyq.pbs
```

## Settings

Use the `chai1` block in the run settings:

```json
"chai1": {
  "gpu_device": 0,
  "device": "cuda:0",
  "target_sequences": null,
  "model_dir": "/scratch/xz2/chai-1_weights/",
  "offline": true,
  "use_esm_embeddings": true,
  "num_trunk_recycles": 3,
  "num_diffn_timesteps": 200,
  "num_diffn_samples": 5,
  "num_trunk_samples": 1,
  "low_memory": true,
  "seed": 42,
  "max_sequences_per_backbone": null
}
```

The script forces `use_msa_server=false`, `use_templates_server=false`, and
`fasta_names_as_cif_chains=true` so predictions remain offline and output chains
match the configured target and binder chain IDs. It points `CHAI_DOWNLOADS_DIR`
at `chai1.model_dir` before importing Chai, so the ESM model is loaded from
`$CHAI_DOWNLOADS_DIR/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`.

## Run

```bash
source /path/to/install/location/chai1_offline/bin/activate
export CHAI_MODEL_DIR=/scratch/xz2/chai-1_weights/

python3 scripts/run_chai1_offline.py --settings settings/my_run.json --limit 100
```

For PBS submission:

```bash
qsub -v VENV=/scratch/xz2/$USER/chai1_offline,CHAI_MODEL_DIR=/scratch/xz2/chai-1_weights/,SETTINGS=/path/to/settings.json,LIMIT=100 scripts/gadi/03_run_chai1_offline.pbs
```
