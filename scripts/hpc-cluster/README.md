# hpc-cluster-3 Slurm jobs

Slurm templates for running the `scaffold_bind_craft` stages on `hpc-cluster-3`:

1. `01_run_colabdesign_rso.sbatch`
2. `02_run_proteinmpnn.sbatch`
3. `03_run_chai1_offline.sbatch`
4. `03_run_fast_protenix.sbatch`
5. `04_run_openmm_screen.sbatch`

Edit the `#SBATCH --mail-user`, partition, and default paths before submitting, or override runtime values with `sbatch --export`.

Example:

```bash
sbatch --export=ALL,REPO_DIR=/scratch/$USER/scaffold_bind_craft,SETTINGS=/scratch/$USER/runs/my_run.json,LIMIT=50 scripts/hpc-cluster/01_run_colabdesign_rso.sbatch
```

Use `OUTPUT_DIR` to override the settings file output directory for a job:

```bash
sbatch --export=ALL,SETTINGS=/scratch/$USER/runs/my_run.json,OUTPUT_DIR=/scratch/$USER/runs/run_001,LIMIT=100 scripts/hpc-cluster/03_run_chai1_offline.sbatch
```

The templates default to `/scratch/$USER/scaffold_bind_craft` for the repo, `/scratch/$USER/logs` for Slurm logs, `/jobfs/$SLURM_JOB_ID` for temporary files, shared environments from `/env/shared` where documented, and user environments under `/envs/$USER` otherwise.

The default partition is `long` because these templates mirror the longer Gadi jobs. For short tests, submit with `--partition=debug --time=04:00:00`. Use `--partition=gpu96` only when the workload needs the 96 GB GPU; for Fast Protenix on `gpu96`, also override the MMseqs module in the script to `mmseqs/18-8cc5c/blackwell`.

Chai-1 runs offline with `CHAI_MODEL_DIR=/models/chai-1` by default. Tune `chai1.model_dir`, `chai1.device`, `chai1.num_trunk_recycles`, `chai1.num_diffn_timesteps`, and `chai1.num_diffn_samples` in the settings JSON for the available GPU memory.
