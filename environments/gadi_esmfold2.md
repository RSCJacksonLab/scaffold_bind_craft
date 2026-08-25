# Gadi ESMFold2 offline environment

ESMFold2 requires Python 3.12. The setup uses Gadi's `python3/3.12.13` module;
prediction jobs load `cuda/12.9.0`. Compute nodes are treated as fully
air-gapped.

## Install code, dependencies, and checkpoints

Submit the network-enabled `copyq` job once:

```bash
qsub environments/00_install_esmfold2_copyq.pbs
```

By default it creates:

- `/scratch/xz2/esmfold2/esm`
- `/scratch/xz2/esmfold2/envs/esmfold2`
- `/scratch/xz2/esmfold2/models/ESMFold2`
- `/scratch/xz2/esmfold2/models/ESMC-6B` when the folding checkpoint does not bundle ESMC
- `/scratch/xz2/esmfold2/huggingface`

The job installs the local ESM checkout and all Python dependencies into the
persistent virtual environment, downloads the complete `biohub/ESMFold2`
snapshot, downloads `biohub/ESMC-6B` if required by that checkpoint, rewrites
the staged folding config to reference the local ESMC directory, verifies the
snapshot contains `ccd.pkl`, and performs an offline input-builder preflight.

Override locations or the source revision with `qsub -v`, for example:

```bash
qsub -v INSTALL_ROOT=/scratch/xz2/$USER/esmfold2,ESM_GIT_REF=main,GROUP=xz2 environments/00_install_esmfold2_copyq.pbs
```

If `/scratch/xz2/esmfold2/esm` already contains the desired
checkout, the job uses it without cloning. This is recommended when the local
checkout contains changes not yet present on GitHub.

## Run offline

Both Gadi design settings point at the staged source and checkpoint. Submit:

```bash
qsub -v SETTINGS=/scratch/xz2/ab8507/scaffold_bind_craft/settings/igg3_gadi.json,LIMIT=100 scripts/gadi/03_run_esmfold2.pbs
qsub -v SETTINGS=/scratch/xz2/ab8507/scaffold_bind_craft/settings/iga_gadi.json,LIMIT=100 scripts/gadi/03_run_esmfold2.pbs
```

The prediction PBS script sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and refuses to start if the local checkpoint is
missing. It passes `<checkpoint>/ccd.pkl` directly to ESMFold2 input builder,
so no model, CCD, or package download is attempted on the GPU node.

The published ESMFold2 implementation uses bfloat16 CUDA operations and a 6B
parameter ESMC backbone. Confirm that the selected Gadi GPU queue provides
bfloat16-capable hardware and sufficient GPU memory; a CUDA module being
available does not by itself guarantee those hardware capabilities.
