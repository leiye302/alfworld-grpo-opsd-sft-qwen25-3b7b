#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON312:-python3.12}"
"$python_bin" -c 'import sys,platform; assert sys.version_info[:2]==(3,12); assert platform.system()=="Linux" and platform.machine()=="x86_64"'
command -v g++ >/dev/null || { echo 'Install g++ before preparing this environment.'; exit 1; }
"$python_bin" -m venv "$root/.venv"
"$root/.venv/bin/python" -m pip install --upgrade 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
"$root/.venv/bin/python" -m pip install -r "$root/configs/bootstrap-requirements.txt"
echo "Next: source $root/.venv/bin/activate"
echo 'Then: python scripts/handoff.py prepare --work /data/alfwork'
