#!/usr/bin/env bash
#SBATCH --job-name=sbc-openmm
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
#SBATCH --mail-user=<YOUR_EMAIL_HERE>
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source /etc/profile
module purge
module load python/3.11.15

REPO_DIR="${REPO_DIR:-$HOME/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/igg3.json}"
LIMIT="${LIMIT:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VENV="${VENV:-$(python3 - "$SETTINGS" <<'PYJSON'
import json
import os
import sys
with open(sys.argv[1]) as handle:
    print(os.path.expandvars(json.load(handle)["hpc"]["venvs"]["openmm"]))
PYJSON
)}"

export TMPDIR="${TMPDIR:-${jobfs:-/jobfs/${SLURM_JOB_ID:-manual}}}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p "/scratch/$USER/logs" "$TMPDIR"

source "$VENV/bin/activate"
cd "$REPO_DIR"

args=(--settings "$SETTINGS" --limit "$LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

python3 scripts/run_openmm_screen.py "${args[@]}"
