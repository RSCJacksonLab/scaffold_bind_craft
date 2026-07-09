#!/usr/bin/env bash
#SBATCH --job-name=sbc-colabdesign
#SBATCH --partition=gpu96
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

source /etc/profile
set -euo pipefail

module purge
module load python/3.10.20

REPO_DIR="${REPO_DIR:-$HOME/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/igg3.json}"
LIMIT="${LIMIT:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VENV="${VENV:-$(python3 - "$SETTINGS" <<'PYJSON'
import json
import sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["tools"]["colabdesign_path"])
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

python3 scripts/run_colabdesign_rso.py "${args[@]}"
