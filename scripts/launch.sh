#!/usr/bin/env bash
# Run inside the user's already selected Linux training server.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="${1:?Usage: bash scripts/launch.sh /your/data-disk/alfworld [--resume]}"
resume_flag="${2:-}"
if [[ $# -gt 2 || ( -n "$resume_flag" && "$resume_flag" != '--resume' ) ]]; then
  echo 'The only optional second argument is --resume.' >&2
  exit 1
fi
command -v tmux >/dev/null || { echo 'Install tmux on this training server first.' >&2; exit 1; }
session_name='alfworld-3b-sft'
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session $session_name already exists; inspect it before launching another job." >&2
  exit 1
fi
mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
command_args=(bash "$repo_root/scripts/run_3b.sh" "$work_dir")
if [[ -n "${PYTHON312:-}" ]]; then
  command_args=(env "PYTHON312=$PYTHON312" "${command_args[@]}")
fi
printf -v run_command '%q ' "${command_args[@]}"
if [[ -n "$resume_flag" ]]; then run_command+=' --resume'; fi
printf -v log_path '%q' "$work_dir/3b_sft_queue.log"
run_command+=" >> $log_path 2>&1"
tmux new-session -d -s "$session_name" -c "$repo_root" "$run_command"
echo "Started environment preparation, downloads and training in tmux: $session_name"
echo "Queue log: $work_dir/3b_sft_queue.log"
echo "Training log: $work_dir/runs/3b_sft/logs/driver.log"
echo 'A tmux session existing does not itself prove that checks or training passed; inspect the logs.'
