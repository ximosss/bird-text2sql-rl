#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
base_model="/data/qwen3-4b-instruct-2507"
base_model_id="bird-text2sql-base"
candidate_model_id="bird-text2sql-sft"
base_eval_config="$project_root/configs/prime-rl/eval/platinum-base-v2.toml"
candidate_eval_config="$project_root/configs/prime-rl/eval/platinum-v2.toml"
run_value="${1:-}"
step="${2:-1965}"

if [[ -z "$run_value" ]]; then
  printf 'Usage: %s <run-name-or-absolute-run-dir> [step]\n' "$0" >&2
  exit 2
fi
if [[ "$run_value" = /* ]]; then
  run_dir="$run_value"
else
  run_dir="$output_root/$run_value"
fi
adapter="$run_dir/adapter"
run_name="$(basename "$run_dir")"
session_name="${BIRD_TMUX_SESSION:-bird-sft-post-eval-$run_name}"
models_url="http://127.0.0.1:8000/v1/models"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
num_tasks_override="${BIRD_EVAL_NUM_TASKS_OVERRIDE:-}"

eval_task_arg=""
if [[ -n "$num_tasks_override" ]]; then
  if [[ ! "$num_tasks_override" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BIRD_EVAL_NUM_TASKS_OVERRIDE must be a positive integer.\n' >&2
    exit 2
  fi
  printf -v eval_task_arg ' --num-tasks %q' "$num_tasks_override"
fi

for required in \
  "$prime_rl_root" \
  "$adapter/adapter_model.safetensors" \
  "$adapter/adapter_config.json" \
  "$base_eval_config" \
  "$candidate_eval_config"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  fi
done
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
    python3 -c 'import json,sys; ids={x["id"] for x in json.load(sys.stdin)["data"]}; raise SystemExit(0 if set(sys.argv[1:]) <= ids else 1)' \
      "$base_model_id" "$candidate_model_id" <<<"$payload"; then
    break
  fi
  sleep 2
done
if ((SECONDS >= deadline)); then
  printf 'Timed out waiting for the eval server.\n' >&2
  exit 1
fi

batch_id="$(date +%Y%m%d-%H%M%S)"
printf -v eval_command \
  'set -o pipefail; env UV_CACHE_DIR=%q prime env install bird-text2sql --path %q --plain && env UV_CACHE_DIR=%q LOCAL_API_KEY=local NO_PROXY=%q no_proxy=%q uv run --no-sync eval @ %q --model %q --run.name %q%s && env UV_CACHE_DIR=%q LOCAL_API_KEY=local NO_PROXY=%q no_proxy=%q uv run --no-sync eval @ %q --model %q --run.name %q%s; status=$?; tmux send-keys -t %q C-c; exit $status' \
  "$uv_cache_dir" "$project_root/environments" \
  "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$base_eval_config" \
  "$base_model_id" "bird-sft-post-base--$batch_id" "$eval_task_arg" \
  "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$candidate_eval_config" \
  "$candidate_model_id" "bird-sft-post-step$step--$batch_id" "$eval_task_arg" "$session_name:server"
tmux new-window -d -t "$session_name:" -n eval -c "$prime_rl_root" "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started post-training eval in tmux session %s\n' "$session_name"
printf 'Base run: bird-sft-post-base--%s\n' "$batch_id"
printf 'Candidate run: bird-sft-post-step%s--%s\n' "$step" "$batch_id"
printf 'Results: %s\n' "$output_root"
