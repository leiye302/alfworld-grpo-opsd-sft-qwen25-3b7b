#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export PIP_CACHE_DIR="$root/.bootstrap_cache"
export TMPDIR="$root/.bootstrap_tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python_bin="${PYTHON312:-python3.12}"
if [[ -z "${PYTHON312:-}" ]] && ! command -v "$python_bin" >/dev/null; then
  python_bin=python3
fi
"$python_bin" -c 'import sys,platform; assert sys.version_info[:2]==(3,12); assert platform.system()=="Linux" and platform.machine()=="x86_64"'
command -v g++ >/dev/null || { echo 'Install g++ before preparing this environment.'; exit 1; }
"$python_bin" -m venv "$root/.venv"
"$root/.venv/bin/python" -m pip install --upgrade 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
"$root/.venv/bin/python" -m pip install -r "$root/configs/bootstrap-requirements.txt" -c "$root/configs/bootstrap-constraints.txt"
echo "Next: source $root/.venv/bin/activate"
echo 'Then: python scripts/handoff.py prepare --work /data/alfwork'
