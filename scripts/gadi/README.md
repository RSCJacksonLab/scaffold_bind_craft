# Gadi PBS jobs

PBS templates for running the four `scaffold_bind_craft` stages on Gadi:

1. `01_run_colabdesign_rso.pbs`
2. `02_run_proteinmpnn.pbs`
3. `03_run_chai1_offline.pbs`
4. `04_run_openmm_screen.pbs`

Edit the `#PBS -P`, `#PBS -M`, storage flags, and default paths before submitting, or override runtime values with `qsub -v`.

Example:

```bash
qsub -v REPO_DIR=/scratch/xz2/$USER/scaffold_bind_craft,SETTINGS=/scratch/xz2/$USER/runs/my_run.json,LIMIT=50 scripts/gadi/01_run_colabdesign_rso.pbs
```

Use `OUTPUT_DIR` to override the settings file output directory for a job:

```bash
qsub -v SETTINGS=/scratch/xz2/$USER/runs/my_run.json,OUTPUT_DIR=/scratch/xz2/$USER/runs/run_001,LIMIT=100 scripts/gadi/03_run_chai1_offline.pbs
```

The Chai-1 job runs in the active `VENV`, which should provide `chai_lab` from `git@github.com:RSCJacksonLab/chai-1-offline.git`, PyTorch, CUDA, and Biopython. Tune `chai1.device`, `chai1.num_trunk_recycles`, `chai1.num_diffn_timesteps`, and `chai1.num_diffn_samples` in the settings JSON for the available GPU memory.

Prime Chai model weights in `CHAI_DOWNLOADS_DIR` from a network-enabled job before offline prediction jobs:

```bash
export CHAI_DOWNLOADS_DIR=/scratch/xz2/$USER/chai_downloads
python -c 'from pathlib import Path; from chai_lab.chai1 import run_inference; f=Path("/tmp/chai_cache.fasta"); f.write_text(">protein|name=A\nGAGAGAGAGAGAGAGA\n"); o=Path("/tmp/chai_cache"); o.mkdir(exist_ok=True); run_inference(fasta_file=f, output_dir=o, use_esm_embeddings=True, use_msa_server=False, use_templates_server=False, num_diffn_samples=1, num_diffn_timesteps=10, device="cuda:0", fasta_names_as_cif_chains=True)'
```

Set the same `CHAI_DOWNLOADS_DIR` for `03_run_chai1_offline.pbs`.
