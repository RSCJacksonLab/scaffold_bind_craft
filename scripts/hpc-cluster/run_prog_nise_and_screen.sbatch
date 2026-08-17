#!/usr/bin/env bash
#SBATCH --job-name=prog-nise-screen
#SBATCH --partition=debug
#SBATCH --nodelist=compute-02
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --tmp=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

source /etc/profile
set -euo pipefail

module purge
module load python/3.10.20

REPO_DIR="${REPO_DIR:-$HOME/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/prog_nise.json}"
MPNN_LIMIT="${MPNN_LIMIT:-1}"
CHAI_LIMIT="${CHAI_LIMIT:-10000}"
NISE_LIMIT="${NISE_LIMIT:-10000}"
SCREEN_LIMIT="${SCREEN_LIMIT:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
CHAI_VENV="${CHAI_VENV:-$(python3 - "$SETTINGS" <<'PYJSON'
import json
import os
import sys
with open(sys.argv[1]) as handle:
    print(os.path.expandvars(json.load(handle)["hpc"]["venvs"]["chai1"]))
PYJSON
)}"

export TMPDIR="${TMPDIR:-${jobfs:-/jobfs/${SLURM_JOB_ID:-manual}}}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/xdg_cache}"
mkdir -p "/scratch/$USER/logs" "$TMPDIR" "$XDG_CACHE_HOME"
if [[ -n "${CHAI_MODEL_DIR:-}" ]]; then
  export CHAI_MODEL_DIR
fi

cd "$REPO_DIR"

module purge
module load python/3.11.15
PROTEINMPNN_VENV="${PROTEINMPNN_VENV:-/cluster/home/anthony/pmpnn}"
source "$PROTEINMPNN_VENV/bin/activate"
mpnn_args=(--settings "$SETTINGS" --limit "$MPNN_LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  mpnn_args+=(--output-dir "$OUTPUT_DIR")
fi
echo "Stage 1/4: initial ProteinMPNN library"
MPNN_STATUS=0
python3 scripts/run_proteinmpnn.py "${mpnn_args[@]}" || MPNN_STATUS=$?
deactivate

module purge
module load python/3.10.20
source "$CHAI_VENV/bin/activate"
chai_args=(--settings "$SETTINGS" --limit "$CHAI_LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  chai_args+=(--output-dir "$OUTPUT_DIR")
fi
echo "Stage 2/4: Chai-1 prediction and confidence screening"
CHAI_STATUS=0
python3 scripts/run_chai1_offline.py "${chai_args[@]}" || CHAI_STATUS=$?

nise_args=(--settings "$SETTINGS" --limit "$NISE_LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  nise_args+=(--output-dir "$OUTPUT_DIR")
fi
echo "Stage 3/4: interface-position NISE"
NISE_STATUS=0
python3 scripts/run_chai1_nise.py "${nise_args[@]}" || NISE_STATUS=$?

deactivate
module purge
module load python/3.11.15
OPENMM_VENV="${OPENMM_VENV:-/envs/shared/openmm}"
source "$OPENMM_VENV/bin/activate"

screen_args=(--settings "$SETTINGS" --limit "$SCREEN_LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  screen_args+=(--output-dir "$OUTPUT_DIR")
fi
echo "Stage 4/4: OpenMM interface screening of NISE predictions"
SCREEN_STATUS=0
python3 scripts/run_openmm_screen.py "${screen_args[@]}" || SCREEN_STATUS=$?

if (( MPNN_STATUS != 0 || CHAI_STATUS != 0 || NISE_STATUS != 0 || SCREEN_STATUS != 0 )); then
  echo "Workflow completed with ProteinMPNN status $MPNN_STATUS, initial Chai status $CHAI_STATUS, NISE status $NISE_STATUS, and screening status $SCREEN_STATUS" >&2
  exit 1
fi
