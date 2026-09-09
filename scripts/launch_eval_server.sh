#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
base_model="${BIRD_EVAL_BASE_MODEL:-/data/qwen3-4b-instruct-2507}"
training_run="${BIRD_EVAL_TRAINING_RUN:-$project_root/outputs/prime-rl/bird-sqlonly-online-20260829-191739-24054}"
best_step="${BIRD_EVAL_BEST_STEP:-80}"
final_step="${BIRD_EVAL_FINAL_STEP:-120}"
best_adapter="${BIRD_EVAL_BEST_ADAPTER:-$training_run/eval-adapters/step_$best_step}"
final_adapter="${BIRD_EVAL_FINAL_ADAPTER:-$training_run/broadcasts/step_$final_step}"
base_model_id="bird-text2sql-base"
best_model_id="bird-text2sql-rl-step$best_step"
final_model_id="bird-text2sql-rl-step$final_step"
models_url="http://127.0.0.1:8000/v1/models"
session_name="${BIRD_TMUX_SESSION:-bird-rl-v1}"
window_name="${BIRD_EVAL_SERVER_WINDOW:-eval-server}"
ready_timeout=600
model_payload=""

merge_no_proxy() {
  local value="${NO_PROXY:-${no_proxy:-}}"
  local host
  for host in localhost 127.0.0.1 127.0.1.1 ::1; do
    case ",$value," in
      *",$host,"*) ;;
      *) value="${value:+$value,}$host" ;;
    esac
  done
  printf '%s' "$value"
}

probe_models() {
  model_payload=""
  if ! model_payload="$(curl --noproxy '*' --silent --show-error --fail --max-time 10 "$models_url" 2>/dev/null)"; then
    return 2
  fi
  python3 -c '
import json
import sys

expected = set(sys.argv[1:])
payload = json.load(sys.stdin)
observed = {str(item.get("id")) for item in payload.get("data", [])}
raise SystemExit(0 if expected <= observed else 1)
' "$base_model_id" "$best_model_id" "$final_model_id" <<<"$model_payload"
}

print_observed_models() {
  python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
    model_ids = [str(item.get("id")) for item in payload.get("data", [])]
except Exception:
    model_ids = []
print(", ".join(model_ids) if model_ids else "<none>")
' <<<"$model_payload"
}

no_proxy_value="$(merge_no_proxy)"
export NO_PROXY="$no_proxy_value"
export no_proxy="$no_proxy_value"
export LOCAL_API_KEY="${LOCAL_API_KEY:-local}"

case "${1:-start}" in
  check)
    if probe_models; then
      printf 'Eval endpoint is ready: %s\n' "$(print_observed_models)"
      exit 0
    fi
    status=$?
    if [[ $status -eq 1 ]]; then
      printf 'Eval endpoint returned unexpected model(s): %s\n' "$(print_observed_models)" >&2
    else
      printf 'Eval endpoint is not reachable: %s\n' "$models_url" >&2
    fi
    exit "$status"
    ;;
  start) ;;
  *)
    printf 'Usage: %s [start|check]\n' "$0" >&2
    exit 2
    ;;
esac

for command_name in curl python3 rg tmux uv; do
  command -v "$command_name" >/dev/null
done
for required_path in "$prime_rl_root" "$base_model" "$best_adapter/adapter_model.safetensors" "$final_adapter/adapter_model.safetensors"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'Required path is missing: %s\n' "$required_path" >&2
    exit 1
  fi
done
if tmux has-session -t "$session_name" 2>/dev/null && \
   tmux list-windows -t "$session_name" -F '#{window_name}' | rg -Fxq "$window_name"; then
  printf 'tmux window already exists: %s:%s\n' "$session_name" "$window_name" >&2
  exit 1
fi
if probe_models; then
  printf 'Eval endpoint is already ready: %s\n' "$(print_observed_models)"
  exit 0
else
  probe_status=$?
fi
if [[ $probe_status -eq 1 ]]; then
  printf 'Port 8000 serves unexpected model(s): %s\n' "$(print_observed_models)" >&2
  exit 1
fi

printf -v server_command \
  'exec env CUDA_VISIBLE_DEVICES=0 NO_PROXY=%q no_proxy=%q uv run --no-sync vllm serve %q --host 127.0.0.1 --port 8000 --served-model-name %q --enable-lora --max-lora-rank 32 --lora-modules %q %q --max-model-len 36864 --gpu-memory-utilization 0.9 --generation-config vllm' \
  "$NO_PROXY" "$no_proxy" "$base_model" \
  "$base_model_id" "$best_model_id=$best_adapter" "$final_model_id=$final_adapter"
if tmux has-session -t "$session_name" 2>/dev/null; then
  tmux new-window -d -t "$session_name:" -n "$window_name" -c "$prime_rl_root" "$server_command"
else
  tmux new-session -d -s "$session_name" -n "$window_name" -c "$prime_rl_root" "$server_command"
fi
tmux set-window-option -t "$session_name:$window_name" remain-on-exit on >/dev/null
printf 'Started tmux window %s:%s; waiting for the direct model check.\n' "$session_name" "$window_name"
printf 'LoRA mapping: %s=%s; %s=%s\n' \
  "$best_model_id" "$best_adapter" "$final_model_id" "$final_adapter"

deadline=$((SECONDS + ready_timeout))
while ((SECONDS < deadline)); do
  if ! tmux list-panes -t "$session_name:$window_name" -F '#{pane_dead}' 2>/dev/null | rg -Fxq 0; then
    printf 'Eval server exited before becoming ready.\n' >&2
    tmux capture-pane -p -t "$session_name:$window_name" -S -80 2>/dev/null || true
    exit 1
  fi
  if probe_models; then
    printf 'Eval endpoint is ready: %s\n' "$(print_observed_models)"
    exit 0
  fi
  sleep 2
done

printf 'Timed out after %ss waiting for %s\n' "$ready_timeout" "$models_url" >&2
exit 1
