# Gadi Chai-1 Offline Environment

This setup installs `git@github.com:RSCJacksonLab/chai-1-offline.git` for
`scripts/run_chai1_offline.py`. The stage runs in single-sequence mode with ESM
embeddings and does not call MSA or template servers during prediction.

## Requirements

- Linux
- Python 3.10 or later
- CUDA GPU with bfloat16 support
- `CHAI_DOWNLOADS_DIR` on shared storage for cached Chai model weights

## Install

Run this on a Gadi node with internet access:

```bash
module load python3/3.10.4 cuda/12.9.0

cd /scratch/xz2/$USER
git clone git@github.com:RSCJacksonLab/chai-1-offline.git

python3 -m venv /scratch/xz2/$USER/chai1_offline
source /scratch/xz2/$USER/chai1_offline/bin/activate

python -m pip install --upgrade pip
python -m pip install -e /scratch/xz2/$USER/chai-1-offline
python -m pip install biopython
```

Chai downloads weights on first use. Prime the cache from a network-enabled GPU
job and keep the same cache path for later offline jobs:

```bash
export CHAI_DOWNLOADS_DIR=/scratch/xz2/$USER/chai_downloads
mkdir -p "$CHAI_DOWNLOADS_DIR"

python - <<'PY'
from pathlib import Path
from chai_lab.chai1 import run_inference

fasta = Path("/tmp/chai_cache_test.fasta")
fasta.write_text(">protein|name=A\nGAGAGAGAGAGAGAGA\n")
out = Path("/tmp/chai_cache_test")
out.mkdir(exist_ok=True)

run_inference(
    fasta_file=fasta,
    output_dir=out,
    use_esm_embeddings=True,
    use_msa_server=False,
    use_templates_server=False,
    num_trunk_recycles=1,
    num_diffn_timesteps=10,
    num_diffn_samples=1,
    seed=42,
    device="cuda:0",
    fasta_names_as_cif_chains=True,
)
PY
```

## Settings

Use the `chai1` block in the run settings:

```json
"chai1": {
  "gpu_device": 0,
  "device": "cuda:0",
  "target_chains": "A",
  "target_sequences": null,
  "binder_chain": "B",
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
match the configured target and binder chain IDs.

## Run

```bash
source /scratch/xz2/$USER/chai1_offline/bin/activate
export CHAI_DOWNLOADS_DIR=/scratch/xz2/$USER/chai_downloads

python3 scripts/run_chai1_offline.py --settings settings/my_run.json --limit 100
```

For PBS submission:

```bash
qsub -v VENV=/scratch/xz2/$USER/chai1_offline,CHAI_DOWNLOADS_DIR=/scratch/xz2/$USER/chai_downloads,SETTINGS=/path/to/settings.json,LIMIT=100 scripts/gadi/03_run_chai1_offline.pbs
```
