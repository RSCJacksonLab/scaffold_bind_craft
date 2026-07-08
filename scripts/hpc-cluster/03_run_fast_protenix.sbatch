#!/usr/bin/env bash
#SBATCH --job-name=sbc-protenix
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --tmp=80G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
#SBATCH --mail-user=<YOUR_EMAIL_HERE>
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source /etc/profile
module purge
module load python/3.12.8
module load mmseqs/18-8cc5c/generic

REPO_DIR="${REPO_DIR:-/scratch/$USER/scaffold_bind_craft}"
SETTINGS="${SETTINGS:-$REPO_DIR/settings/example.json}"
LIMIT="${LIMIT:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VENV="${VENV:-/envs/$USER/fast_protenix_alphafast}"

export TMPDIR="${TMPDIR:-${jobfs:-/jobfs/$SLURM_JOB_ID}}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export MMSEQS_TMP="${MMSEQS_TMP:-$TMPDIR/mmseqs}"
mkdir -p "/scratch/$USER/logs" "$TMPDIR" "$MMSEQS_TMP"

source "$VENV/bin/activate"
cd "$REPO_DIR"

args=(--settings "$SETTINGS" --limit "$LIMIT")
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

python3 scripts/run_fast_protenix.py "${args[@]}"
