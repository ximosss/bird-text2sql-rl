#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config="$project_root/configs/prime-rl/eval/revisql-platinum-base-smoke.toml"
model="/data/qwen3-4b-instruct-2507"
model_id="bird-revisql-base"
session_name="${BIRD_TMUX_SESSION:-bird-revisql-base-smoke-$(date +%Y%m%d-%H%M%S)}"
run_name="${1:-bird-revisql-base-smoke-$(date +%Y%m%d-%H%M%S)}"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
models_url="http://127.0.0.1:8000/v1/models"

for command_name in curl jq nvidia-smi prime tmux uv; do
  command -v "$command_name" >/dev/null || {
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done
if ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'No NVIDIA GPU is visible from the host namespace.\n' >&2
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
  'exec env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=%q NO_PROXY=%q no_proxy=%q uv run --no-sync vllm serve %q --host 127.0.0.1 --port 8000 --served-model-name %q --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --gpu-memory-utilization 0.9 --generation-config vllm' \
  "$uv_cache_dir" "$no_proxy_value" "$no_proxy_value" "$model" "$model_id"
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
    jq -e --arg model "$model_id" '.data | any(.id == $model)' <<<"$payload" >/dev/null; then
    break
  fi
  sleep 2
done
if ((SECONDS >= deadline)); then
  printf 'Timed out waiting for the eval server.\n' >&2
  exit 1
fi

printf -v eval_command \
  'set -o pipefail; env UV_CACHE_DIR=%q prime env install bird-text2sql --path %q --plain && env UV_CACHE_DIR=%q LOCAL_API_KEY=local NO_PROXY=%q no_proxy=%q uv run --no-sync eval @ %q --run.name %q; status=$?; tmux send-keys -t %q C-c; exit $status' \
  "$uv_cache_dir" "$project_root/environments" "$uv_cache_dir" "$no_proxy_value" \
  "$no_proxy_value" "$config" "$run_name" "$session_name:server"
tmux new-window -d -t "$session_name:" -n eval -c "$prime_rl_root" "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started ReViSQL base smoke eval in tmux session: %s\n' "$session_name"
printf 'Run: %s\n' "$run_name"
