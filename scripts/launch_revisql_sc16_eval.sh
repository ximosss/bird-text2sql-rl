#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
config="$project_root/configs/prime-rl/eval/revisql-arcwise-plat-sql-sc16.toml"
base_model="/data/qwen3-4b-instruct-2507"
adapter="${BIRD_ADAPTER:-$output_root/bird-revisql-qwen3-4b-promptfmt-v4-pilot40-20260913/adapter-step1300}"
model_id="${BIRD_CANDIDATE_MODEL_ID:-bird-revisql-step1300}"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
mode="${1:-formal}"
models_url="http://127.0.0.1:8000/v1/models"

case "$mode" in
  smoke)
    num_tasks="${BIRD_EVAL_NUM_TASKS_OVERRIDE:-2}"
    run_kind="smoke"
    ;;
  formal)
    num_tasks=498
    run_kind="formal"
    ;;
  *)
    printf 'Usage: %s [smoke|formal]\n' "$0" >&2
    exit 2
    ;;
esac
[[ "$num_tasks" =~ ^[1-9][0-9]*$ ]] || {
  printf 'BIRD_EVAL_NUM_TASKS_OVERRIDE must be a positive integer.\n' >&2
  exit 2
}

for command_name in curl jq nvidia-smi prime tmux uv; do
  command -v "$command_name" >/dev/null || {
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done
for required in "$prime_rl_root" "$config" "$base_model/config.json" "$adapter/adapter_model.safetensors" "$adapter/adapter_config.json"; do
  [[ -e "$required" ]] || {
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  }
done
nvidia-smi -L >/dev/null 2>&1 || {
  printf 'No NVIDIA GPU is visible.\n' >&2
  exit 1
}
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

batch_id="$(date +%Y%m%d-%H%M%S)"
run_name="bird-revisql-step1300-arcwise-plat-sql-sc16-$run_kind--$batch_id"
session_name="${BIRD_TMUX_SESSION:-$run_name}"
tmux has-session -t "$session_name" 2>/dev/null && {
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
}

printf -v server_command 'exec env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=%q NO_PROXY=%q no_proxy=%q uv run --no-sync vllm serve %q --host 127.0.0.1 --port 8000 --served-model-name bird-revisql-base --enable-lora --max-lora-rank 32 --lora-modules %q --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --gpu-memory-utilization 0.9 --generation-config vllm' "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$base_model" "$model_id=$adapter"
tmux new-session -d -s "$session_name" -n server -c "$prime_rl_root" "$server_command"
tmux set-window-option -t "$session_name:server" remain-on-exit on >/dev/null

deadline=$((SECONDS + 600))
while ((SECONDS < deadline)); do
  if [[ "$(tmux display-message -p -t "$session_name:server" '#{pane_dead}' 2>/dev/null || true)" == 1 ]]; then
    printf 'Eval server exited before becoming ready.\n' >&2
    tmux capture-pane -p -t "$session_name:server" -S -80 2>/dev/null || true
    exit 1
  fi
  if payload="$(curl --noproxy '*' --silent --fail --max-time 3 "$models_url" 2>/dev/null)" && jq -e --arg model "$model_id" '.data | any(.id == $model)' <<<"$payload" >/dev/null; then
    break
  fi
  sleep 2
done
if ((SECONDS >= deadline)); then
  printf 'Timed out waiting for the eval server.\n' >&2
  exit 1
fi

traces="$output_root/$run_name/traces.jsonl"
summary="$output_root/$run_name/sc-summary.json"
printf -v eval_command 'set -o pipefail; env UV_CACHE_DIR=%q prime env install bird-text2sql --path %q --plain && env UV_CACHE_DIR=%q LOCAL_API_KEY=local NO_PROXY=%q no_proxy=%q uv run --no-sync eval @ %q --model %q --num-tasks %q --run.name %q && env UV_CACHE_DIR=%q uv run %q %q --expected-rollouts 16 --output %q; status=$?; tmux send-keys -t %q C-c; exit $status' "$uv_cache_dir" "$project_root/environments" "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$config" "$model_id" "$num_tasks" "$run_name" "$uv_cache_dir" "$project_root/scripts/summarize_sc.py" "$traces" "$summary" "$session_name:server"

tmux new-window -d -t "$session_name:" -n eval -c "$prime_rl_root" "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started local SC-16 %s evaluation in tmux session %s\n' "$run_kind" "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Summary: %s\n' "$summary"
