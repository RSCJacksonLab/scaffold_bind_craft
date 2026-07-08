#!/usr/bin/env bash
#SBATCH --job-name=sbc-chai1
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --tmp=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
#SBATCH --mail-user=<YOUR_EMAIL_HERE>
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source /etc/profile
module purge
module load python/3.10.20

REPO_DIR="${REPO_DIR:-/scratch/$USER/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/example.json}"
LIMIT="${LIMIT:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VENV="${VENV:-/env/shared/chai1_offline}"
CHAI_MODEL_DIR="${CHAI_MODEL_DIR:-/models/chai-1}"

export TMPDIR="${TMPDIR:-${jobfs:-/jobfs/$SLURM_JOB_ID}}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export CHAI_MODEL_DIR
export CHAI_DOWNLOADS_DIR="$CHAI_MODEL_DIR"
export HF_HOME="$CHAI_MODEL_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/xdg_cache}"
mkdir -p "/scratch/$USER/logs" "$TMPDIR" "$XDG_CACHE_HOME"

source "$VENV/bin/activate"
cd "$REPO_DIR"

args=(--settings "$SETTINGS" --limit "$LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

python3 scripts/run_chai1_offline.py "${args[@]}"
