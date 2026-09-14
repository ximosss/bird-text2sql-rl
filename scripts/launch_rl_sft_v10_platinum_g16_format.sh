#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config="${BIRD_RL_CONFIG:-$project_root/configs/prime-rl/rl-sft-v10-platinum-g16-format-v2.toml}"
output_root="$project_root/outputs/prime-rl"
model="/data/ximo/bird-text2sql-rl/models/qwen3-4b-bird-sft-v10"
data_dir="$project_root/data/processed/rlvr-v2"
mode="${1:-start}"
run_prefix="${BIRD_RL_RUN_PREFIX:-bird-rl-sft-v10-plat-full-g16-format-v2}"
default_steps="${BIRD_RL_DEFAULT_STEPS:-40}"
default_name="$run_prefix-$(date +%Y%m%d-%H%M%S)"
run_name="${2:-$default_name}"
target_steps="${3:-}"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
no_proxy_value="${NO_PROXY:-${no_proxy:-}}"
for host in 127.0.0.1 localhost 0.0.0.0; do
  case ",$no_proxy_value," in
    *",$host,"*) ;;
    *) no_proxy_value="${no_proxy_value:+$no_proxy_value,}$host" ;;
  esac
done

case "$mode" in
  check|smoke|start|resume) ;;
  *)
    printf 'Usage: %s [check|smoke|start|resume] [run-name] [target-steps]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "resume" && $# -lt 2 ]]; then
  printf 'Resume requires an existing run name.\n' >&2
  exit 2
fi

for command_name in nvidia-smi tmux uv sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done
for required in \
  "$prime_rl_root/pyproject.toml" \
  "$config" \
  "$model/config.json" \
  "$model/model.safetensors.index.json" \
  "$data_dir/train.jsonl" \
  "$data_dir/validation.jsonl" \
  "$data_dir/manifest.json"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  fi
done
if [[ "$(wc -l < "$data_dir/train.jsonl")" -ne 2050 ]] || \
   [[ "$(wc -l < "$data_dir/validation.jsonl")" -ne 396 ]]; then
  printf 'RLVR-v2 row counts do not match the 2050/396 contract.\n' >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'No NVIDIA GPU is visible; refusing to start RLVR.\n' >&2
  exit 1
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid run name: %s\n' "$run_name" >&2
  exit 2
fi

if [[ "$mode" == "check" ]]; then
  config_sha="$(sha256sum "$config" | cut -d ' ' -f 1)"
  manifest_sha="$(sha256sum "$data_dir/manifest.json" | cut -d ' ' -f 1)"
  model_sha="$(sha256sum "$model/model.safetensors.index.json" | cut -d ' ' -f 1)"
  printf 'config_sha256=%s\nmanifest_sha256=%s\nmodel_index_sha256=%s\n' \
    "$config_sha" "$manifest_sha" "$model_sha"
  cd "$prime_rl_root"
  exec env UV_CACHE_DIR="$uv_cache_dir" NO_PROXY="$no_proxy_value" no_proxy="$no_proxy_value" \
    uv run --no-sync rl @ "$config" \
    --run.name "$run_name" --dry-run True
fi

if [[ "$mode" == "smoke" ]]; then
  run_name="${2:-$run_prefix-smoke-$(date +%Y%m%d-%H%M%S)}"
  max_steps=2
  extra_args="--orchestrator.eval.num-examples 32"
elif [[ "$mode" == "resume" ]]; then
  max_steps="${target_steps:-$default_steps}"
  extra_args="--resume"
else
  max_steps="${target_steps:-$default_steps}"
  extra_args=""
fi
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Target steps must be a positive integer: %s\n' "$max_steps" >&2
  exit 2
fi

if [[ "$mode" == "resume" ]]; then
  session_name="${BIRD_TMUX_SESSION:-${run_name}-resume-${max_steps}}"
else
  session_name="${BIRD_TMUX_SESSION:-$run_name}"
fi

if [[ "$mode" == "resume" && ! -d "$output_root/$run_name/checkpoints" ]]; then
  printf 'Run has no checkpoint directory to resume: %s\n' "$output_root/$run_name" >&2
  exit 1
elif [[ "$mode" != "resume" && -e "$output_root/$run_name" ]]; then
  printf 'Run directory already exists: %s\n' "$output_root/$run_name" >&2
  exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

printf -v train_command \
  'exec env UV_CACHE_DIR=%q NO_PROXY=%q no_proxy=%q uv run --no-sync rl @ %q --run.name %q --max-steps %q %s' \
  "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$config" "$run_name" "$max_steps" "$extra_args"
tmux new-session -d -s "$session_name" -n train -c "$prime_rl_root" "$train_command"
tmux set-window-option -t "$session_name:train" remain-on-exit on >/dev/null

printf 'Started strict-format RLVR %s in tmux session: %s\n' "$mode" "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Attach: tmux attach -t %s\n' "$session_name"
printf 'Results: %s\n' "$output_root/$run_name"
