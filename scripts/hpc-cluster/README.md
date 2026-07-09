# hpc-cluster-3 Slurm jobs

Slurm templates for running the `scaffold_bind_craft` stages on `hpc-cluster-3`:

1. `01_run_colabdesign_rso.sbatch`
2. `02_run_proteinmpnn.sbatch`
3. `03_run_chai1_offline.sbatch`
4. `03_run_fast_protenix.sbatch`
5. `04_run_openmm_screen.sbatch`

Edit the `#SBATCH --mail-user` and scheduler resources before submitting. Pipeline paths come from the settings JSON, with `REPO_DIR`, `SETTINGS`, `OUTPUT_DIR`, `LIMIT`, and `VENV` still available as `sbatch --export` overrides.

Example:

```bash
sbatch --export=ALL,REPO_DIR=/scratch/$USER/scaffold_bind_craft,SETTINGS=/scratch/$USER/runs/my_run.json,LIMIT=50 scripts/hpc-cluster/01_run_colabdesign_rso.sbatch
```

Use `OUTPUT_DIR` to override the settings file output directory for a job:

```bash
sbatch --export=ALL,SETTINGS=/scratch/$USER/runs/my_run.json,OUTPUT_DIR=/scratch/$USER/runs/run_001,LIMIT=100 scripts/hpc-cluster/03_run_chai1_offline.sbatch
```

The templates default `REPO_DIR` to `$HOME/scaffold_bind_craft` and `SETTINGS` to `$REPO_DIR/settings/igg3.json`. Slurm logs default to `/scratch/$USER/logs`, temporary files to `/jobfs/$SLURM_JOB_ID`, and stage virtual environments to `hpc.venvs` in the selected settings JSON.

The default partition is `long` because these templates mirror the longer Gadi jobs. For short tests, submit with `--partition=debug --time=04:00:00`. Use `--partition=gpu96` only when the workload needs the 96 GB GPU; for Fast Protenix on `gpu96`, also override the MMseqs module in the script to `mmseqs/18-8cc5c/blackwell`.

Chai-1 runs offline by default and uses `chai1.model_dir` from the settings JSON unless `CHAI_MODEL_DIR` is explicitly exported. Tune `chai1.device`, `chai1.num_trunk_recycles`, `chai1.num_diffn_timesteps`, and `chai1.num_diffn_samples` there for the available GPU memory.
