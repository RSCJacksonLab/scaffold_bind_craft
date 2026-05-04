# Gadi PBS jobs

PBS templates for running the four `scaffold_bind_craft` stages on Gadi:

1. `01_run_colabdesign_rso.pbs`
2. `02_run_proteinmpnn.pbs`
3. `03_run_esmfold.pbs`
4. `04_run_openmm_screen.pbs`

Edit the `#PBS -P`, `#PBS -M`, storage flags, and default paths before submitting, or override runtime values with `qsub -v`.

Example:

```bash
qsub -v REPO_DIR=/scratch/xz2/$USER/scaffold_bind_craft,SETTINGS=/scratch/xz2/$USER/runs/my_run.json,LIMIT=50 scripts/gadi/01_run_colabdesign_rso.pbs
```

Use `OUTPUT_DIR` to override the settings file output directory for a job:

```bash
qsub -v SETTINGS=/scratch/xz2/$USER/runs/my_run.json,OUTPUT_DIR=/scratch/xz2/$USER/runs/run_001,LIMIT=100 scripts/gadi/03_run_esmfold.pbs
```

The ESMFold job runs in the active `VENV`, which should provide PyTorch and `fair-esm[esmfold]`. Tune `esmfold.gpu_device`, `esmfold.chunk_size`, and `esmfold.num_recycles` in the settings JSON for the available GPU memory.
