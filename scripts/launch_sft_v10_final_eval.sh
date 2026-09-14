#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
base_model="/data/qwen3-4b-instruct-2507"
adapter="$output_root/bird-sft-cot-sql-v10-20260906-0454/adapter"
base_model_id="bird-text2sql-base"
candidate_model_id="bird-text2sql-sft-v10"
session_name="${BIRD_TMUX_SESSION:-bird-sft-v10-final-eval-$(date +%Y%m%d-%H%M%S)}"
models_url="http://127.0.0.1:8000/v1/models"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
num_tasks_override="${BIRD_EVAL_NUM_TASKS_OVERRIDE:-}"
run_prefix="${BIRD_RUN_PREFIX:-bird-sft-v10}"

declare -a benchmark_names=(bird-mini-dev bird-full-dev arcwise-plat-sql arcwise-plat)
if (($#)); then
  benchmark_names=("$@")
fi

declare -a configs=()
for benchmark_name in "${benchmark_names[@]}"; do
  case "$benchmark_name" in
    bird-mini-dev|bird-full-dev|arcwise-plat-sql|arcwise-plat)
      configs+=("$project_root/configs/prime-rl/eval/$benchmark_name.toml")
      ;;
    *)
      printf 'Unknown benchmark: %s\n' "$benchmark_name" >&2
      printf 'Supported benchmarks: bird-mini-dev bird-full-dev arcwise-plat-sql arcwise-plat\n' >&2
      exit 2
      ;;
  esac
done

eval_task_arg=""
if [[ -n "$num_tasks_override" ]]; then
  if [[ ! "$num_tasks_override" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BIRD_EVAL_NUM_TASKS_OVERRIDE must be a positive integer.\n' >&2
    exit 2
  fi
  printf -v eval_task_arg ' --num-tasks %q' "$num_tasks_override"
fi

for command_name in curl jq nvidia-smi prime tmux uv; do
  if ! command -v "$command_name" >/dev/null; then
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done
for required in \
  "$prime_rl_root" \
  "$base_model" \
  "$adapter/adapter_model.safetensors" \
  "$adapter/adapter_config.json" \
  "${configs[@]}"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  fi
done
if ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'No NVIDIA GPU is visible; refusing to start the evaluation.\n' >&2
  exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi
if curl --noproxy '*' --silent --fail --max-time 3 "$models_url" >/dev/null 2>&1; then
  printf 'Port 8000 is already occupied by a model server.\n' >&2
  exit 1
fi

no_proxy_value="${NO_PROXY:-${no_proxy:-}}"
for host in localhost 127.0.0.1 127.0.1.1 ::1; do
  case ",$no_proxy_value," in
    *",$host,"*) ;;
    *) no_proxy_value="${no_proxy_value:+$no_proxy_value,}$host" ;;
  esac
done

printf -v server_command \
  'exec env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=%q NO_PROXY=%q no_proxy=%q uv run --no-sync vllm serve %q --host 127.0.0.1 --port 8000 --served-model-name %q --enable-lora --max-lora-rank 32 --lora-modules %q --reasoning-parser qwen3 --max-model-len 36864 --gpu-memory-utilization 0.9 --generation-config vllm' \
  "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$base_model" \
  "$base_model_id" "$candidate_model_id=$adapter"
tmux new-session -d -s "$session_name" -n server -c "$prime_rl_root" "$server_command"
tmux set-window-option -t "$session_name:server" remain-on-exit on >/dev/null

deadline=$((SECONDS + 600))
while ((SECONDS < deadline)); do
  if [[ "$(tmux display-message -p -t "$session_name:server" '#{pane_dead}' 2>/dev/null || true)" == 1 ]]; then
    printf 'Eval server exited before becoming ready.\n' >&2
    tmux capture-pane -p -t "$session_name:server" -S -80 2>/dev/null || true
    exit 1
  fi
  if payload="$(curl --noproxy '*' --silent --fail --max-time 3 "$models_url" 2>/dev/null)" && \
    jq -e --arg model "$candidate_model_id" '.data | any(.id == $model)' <<<"$payload" >/dev/null; then
    break
  fi
  sleep 2
done
if ((SECONDS >= deadline)); then
  printf 'Timed out waiting for the eval server.\n' >&2
  exit 1
fi

batch_id="$(date +%Y%m%d-%H%M%S)"
eval_command="set -o pipefail; env UV_CACHE_DIR=$(printf %q "$uv_cache_dir") prime env install bird-text2sql --path $(printf %q "$project_root/environments") --plain"
for index in "${!configs[@]}"; do
  run_name="$run_prefix-${benchmark_names[$index]}--$batch_id"
  printf -v eval_command \
    '%s && env UV_CACHE_DIR=%q LOCAL_API_KEY=local NO_PROXY=%q no_proxy=%q uv run --no-sync eval @ %q --model %q --run.name %q%s' \
    "$eval_command" "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" \
    "${configs[$index]}" "$candidate_model_id" "$run_name" "$eval_task_arg"
done
printf -v eval_command '%s; status=$?; tmux send-keys -t %q C-c; exit $status' \
  "$eval_command" "$session_name:server"

tmux new-window -d -t "$session_name:" -n eval -c "$prime_rl_root" "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started SFT v10 final evals in tmux session %s\n' "$session_name"
for benchmark_name in "${benchmark_names[@]}"; do
  printf 'Run: %s-%s--%s\n' "$run_prefix" "$benchmark_name" "$batch_id"
done
printf 'Results: %s\n' "$output_root"
