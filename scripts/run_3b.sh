#!/usr/bin/env bash
# The entire pipeline is inside tmux, including installation and downloads.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="${1:?A data-directory argument is required.}"
resume_flag="${2:-}"
if [[ $# -gt 2 || ( -n "$resume_flag" && "$resume_flag" != '--resume' ) ]]; then
  echo 'Usage: bash scripts/run_3b.sh /your/data-disk/alfworld [--resume]' >&2
  exit 1
fi
echo "Preparing 3B SFT in $work_dir"
bash "$repo_root/scripts/bootstrap.sh"
python_bin="$repo_root/.venv/bin/python"
"$python_bin" -u "$repo_root/scripts/handoff.py" prepare --work "$work_dir" --size 3b
args=(run --work "$work_dir" --size 3b --method sft)
if [[ -n "$resume_flag" ]]; then args+=(--resume); fi
"$python_bin" -u "$repo_root/scripts/handoff.py" "${args[@]}"
