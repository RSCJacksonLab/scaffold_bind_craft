# Gadi PBS jobs

PBS templates for running the four `scaffold_bind_craft` stages on Gadi:

1. `01_run_colabdesign_rso.pbs`
2. `02_run_proteinmpnn.pbs`
3. `03_run_fast_protenix.pbs`
4. `04_run_openmm_screen.pbs`

Edit the `#PBS -P`, `#PBS -M`, storage flags, and default paths before submitting, or override runtime values with `qsub -v`.

Example:

```bash
qsub -v REPO_DIR=/scratch/xz2/$USER/scaffold_bind_craft,SETTINGS=/scratch/xz2/$USER/runs/my_run.json,LIMIT=50 scripts/gadi/01_run_colabdesign_rso.pbs
```

Use `OUTPUT_DIR` to override the settings file output directory for a job:

```bash
qsub -v SETTINGS=/scratch/xz2/$USER/runs/my_run.json,OUTPUT_DIR=/scratch/xz2/$USER/runs/run_001,LIMIT=100 scripts/gadi/03_run_fast_protenix.pbs
```

The Fast Protenix job loads the if89 `MMseqs2/18-8cc5c` module so `mmseqs` is available, and uses `$PBS_JOBFS` for temporary files. Set `VENV` to the AlphaFast / Fast Protenix environment. Protenix can live in a separate environment as long as `fast_protenix.protenix_bin` in the settings JSON points to that environment's `protenix` executable. The database locations still come from the `fast_protenix.db_dir`, `fast_protenix.mmseqs_db_dir`, and `fast_protenix.alphafast_dir` keys.
