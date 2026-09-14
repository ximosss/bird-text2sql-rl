#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config="$project_root/configs/prime-rl/rl-revisql-bird-qwen3-4b-v1.toml"
output_root="$project_root/outputs/prime-rl"
model="/data/qwen3-4b-instruct-2507"
data_dir="$project_root/data/processed/rlvr-v2"
verieql_path="$project_root/third_party/VeriEQL"
mode="${1:-check}"
run_prefix="bird-revisql-qwen3-4b-v1"
run_name="${2:-$run_prefix-$(date +%Y%m%d-%H%M%S)}"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"

case "$mode" in
  check|smoke|pilot|resume) ;;
  *)
    printf 'Usage: %s [check|smoke|pilot|resume] [run-name] [target-steps]\n' "$0" >&2
    exit 2
    ;;
esac
if [[ "$mode" == "resume" && $# -lt 2 ]]; then
  printf 'Resume requires an existing run name.\n' >&2
  exit 2
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid run name: %s\n' "$run_name" >&2
  exit 2
fi

for command_name in git nvidia-smi prime sha256sum tmux uv; do
  command -v "$command_name" >/dev/null || {
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done
for required in \
  "$prime_rl_root/pyproject.toml" \
  "$config" \
  "$model/config.json" \
  "$model/model.safetensors.index.json" \
  "$data_dir/train.jsonl" \
  "$data_dir/validation.jsonl" \
  "$data_dir/manifest.json" \
  "$verieql_path/environment.py"; do
  [[ -e "$required" ]] || {
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  }
done
if [[ "$(wc -l < "$data_dir/train.jsonl")" -ne 2050 ]] || \
   [[ "$(wc -l < "$data_dir/validation.jsonl")" -ne 396 ]]; then
  printf 'Adapted executable split does not match the audited 2050/396 manifest.\n' >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'No NVIDIA GPU is visible from the host namespace.\n' >&2
  exit 1
fi

no_proxy_value="${NO_PROXY:-${no_proxy:-}}"
for host in localhost 127.0.0.1 127.0.1.1 ::1 0.0.0.0; do
  case ",$no_proxy_value," in
    *",$host,"*) ;;
    *) no_proxy_value="${no_proxy_value:+$no_proxy_value,}$host" ;;
  esac
done

preflight_record_dir="$output_root/preflight"
preflight_record="$preflight_record_dir/revisql-bird-qwen3-4b-v1.txt"
config_sha="$(sha256sum "$config" | cut -d ' ' -f 1)"
manifest_sha="$(sha256sum "$data_dir/manifest.json" | cut -d ' ' -f 1)"
model_sha="$(sha256sum "$model/model.safetensors.index.json" | cut -d ' ' -f 1)"
code_sha="$(find "$project_root/environments/bird_text2sql/bird_text2sql" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1)"
printf -v preflight_identity \
  'commit=%s\ncode_sha256=%s\nconfig_sha256=%s\nmanifest_sha256=%s\nmodel_index_sha256=%s' \
  "$(git -C "$project_root" rev-parse HEAD)" "$code_sha" "$config_sha" "$manifest_sha" "$model_sha"

if [[ "$mode" == "check" ]]; then
  env UV_CACHE_DIR="$uv_cache_dir" prime env install bird-text2sql \
    --path "$project_root/environments" --plain
  env UV_CACHE_DIR="$uv_cache_dir" VERIEQL_PATH="$verieql_path" \
    uv run pytest "$project_root/environments/bird_text2sql/tests" -q
  (
    cd "$prime_rl_root"
    env UV_CACHE_DIR="$uv_cache_dir" uv run --no-sync pytest \
      "$project_root/tests/test_cispo.py" -q
    env UV_CACHE_DIR="$uv_cache_dir" uv run --no-sync python \
      "$project_root/scripts/check_revisql_renderer.py" "$model"
  )
  env UV_CACHE_DIR="$uv_cache_dir" VERIEQL_PATH="$verieql_path" \
    uv run "$project_root/scripts/check_revisql_verieql.py"
  preflight_dir="/tmp/bird-revisql-preflight"
  (
    cd "$prime_rl_root"
    env UV_CACHE_DIR="$uv_cache_dir" VERIEQL_PATH="$verieql_path" \
      NO_PROXY="$no_proxy_value" no_proxy="$no_proxy_value" \
      uv run --no-sync rl @ "$config" --output-dir "$preflight_dir" \
      --run.name revisql-bird-qwen3-4b-v1 --dry-run True
  )
  mkdir -p "$preflight_record_dir"
  printf '%s\n' "$preflight_identity" | tee "$preflight_record"
  printf 'preflight_record=%s\n' "$preflight_record"
  exit 0
fi

if [[ ! -f "$preflight_record" ]] || [[ "$(<"$preflight_record")" != "$preflight_identity" ]]; then
  printf 'Preflight identity is missing or stale for this version. Run: %s check\n' "$0" >&2
  exit 1
fi

case "$mode" in
  smoke)
    max_steps=2
    extra_args=(
      --orchestrator.eval.skip-first-step True
      --orchestrator.eval.num-examples 8
    )
    ;;
  pilot)
    max_steps=40
    extra_args=()
    ;;
  resume)
    max_steps="${3:-40}"
    extra_args=(--resume)
    ;;
esac
[[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || {
  printf 'Target steps must be a positive integer: %s\n' "$max_steps" >&2
  exit 2
}

session_name="${BIRD_TMUX_SESSION:-$run_name}"
if [[ "$mode" == "resume" ]]; then
  [[ -d "$output_root/$run_name/checkpoints" ]] || {
    printf 'Run has no checkpoint directory to resume: %s\n' "$output_root/$run_name" >&2
    exit 1
  }
  session_name="${BIRD_TMUX_SESSION:-$run_name-resume-$max_steps}"
elif [[ -e "$output_root/$run_name" ]]; then
  printf 'Run directory already exists: %s\n' "$output_root/$run_name" >&2
  exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

printf -v train_command \
  'exec env UV_CACHE_DIR=%q VERIEQL_PATH=%q NO_PROXY=%q no_proxy=%q uv run --no-sync rl @ %q --run.name %q --max-steps %q' \
  "$uv_cache_dir" "$verieql_path" "$no_proxy_value" "$no_proxy_value" \
  "$config" "$run_name" "$max_steps"
for arg in "${extra_args[@]}"; do
  printf -v train_command '%s %q' "$train_command" "$arg"
done
tmux new-session -d -s "$session_name" -n train -c "$prime_rl_root" "$train_command"
tmux set-window-option -t "$session_name:train" remain-on-exit on >/dev/null

printf 'Started ReViSQL-BIRD %s in tmux session: %s\n' "$mode" "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Attach: tmux attach -t %s\n' "$session_name"
printf 'Results: %s\n' "$output_root/$run_name"
