#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config="$project_root/configs/prime-rl/sft-bird.toml"
output_root="$project_root/outputs/prime-rl"
base_model="/data/qwen3-4b-instruct-2507"
data_dir="$project_root/data/processed/sft-bird-cot-sql-v1"
run_name="${1:-bird-sft-cot-sql-$(date +%Y%m%d-%H%M%S)}"
session_name="${BIRD_TMUX_SESSION:-$run_name}"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"

for command_name in nvidia-smi tmux uv; do
  if ! command -v "$command_name" >/dev/null; then
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done
for required in \
  "$prime_rl_root/pyproject.toml" \
  "$config" \
  "$base_model/config.json" \
  "$data_dir/train.jsonl" \
  "$data_dir/validation.jsonl" \
  "$data_dir/manifest.json"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  fi
done
if ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'No NVIDIA GPU is visible; refusing to start SFT.\n' >&2
  exit 1
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid run name: %s\n' "$run_name" >&2
  exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid tmux session name: %s\n' "$session_name" >&2
  exit 2
fi
if [[ -e "$output_root/$run_name" ]]; then
  printf 'Run directory already exists: %s\n' "$output_root/$run_name" >&2
  exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

printf -v train_command \
  'exec env UV_CACHE_DIR=%q uv run --no-sync sft @ %q --run.name %q' \
  "$uv_cache_dir" "$config" "$run_name"
tmux new-session -d -s "$session_name" -n train -c "$prime_rl_root" "$train_command"
tmux set-window-option -t "$session_name:train" remain-on-exit on >/dev/null

printf 'Started SFT in tmux session: %s\n' "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Attach: tmux attach -t %s\n' "$session_name"
printf 'Results: %s\n' "$output_root/$run_name"
