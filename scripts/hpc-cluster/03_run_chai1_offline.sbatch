#!/usr/bin/env bash
#SBATCH --job-name=sbc-chai1
#SBATCH --partition=gpu96
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --tmp=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
#SBATCH --mail-user=<YOUR_EMAIL_HERE>
#SBATCH --mail-type=FAIL,END

source /etc/profile
set -euo pipefail

module purge
module load python/3.10.20

REPO_DIR="${REPO_DIR:-$HOME/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/igg3.json}"
LIMIT="${LIMIT:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VENV="${VENV:-$(python3 - "$SETTINGS" <<'PYJSON'
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

source "$VENV/bin/activate"
cd "$REPO_DIR"

args=(--settings "$SETTINGS" --limit "$LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

python3 scripts/run_chai1_offline.py "${args[@]}"
