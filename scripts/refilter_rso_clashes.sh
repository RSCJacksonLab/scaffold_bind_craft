#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif python3 -c 'import Bio' >/dev/null 2>&1; then
  python_bin=python3
else
  python_bin=python
fi
exec "$python_bin" "$script_dir/refilter_rso_clashes.py" "$@"
