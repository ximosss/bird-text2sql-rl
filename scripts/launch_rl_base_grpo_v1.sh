#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config_path="${PRIME_RL_CONFIG:-$project_root/configs/prime-rl/rl-base-grpo-v1.toml}"
session_name="${PRIME_RL_TMUX_SESSION:-bird-rl-v1}"
models_url="${PRIME_RL_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
expected_model="${PRIME_RL_EXPECTED_MODEL:-/data/qwen3-4b-instruct-2507}"
ready_timeout="${PRIME_RL_READY_TIMEOUT:-1800}"
run_name="${PRIME_RL_RUN_NAME:-bird-sqlonly-online-$(date +%Y%m%d-%H%M%S)-$RANDOM}"
output_root="$project_root/outputs/prime-rl"
dashboard_url="${PRIME_RL_DASHBOARD_URL:-http://127.0.0.1:7788}"
startup_log="$output_root/startup-$run_name.log"
model_payload=""

usage() {
  printf 'Usage: %s [start|check]\n' "$0" >&2
  printf '  start  start training and the local dashboard in one tmux session\n' >&2
  printf '  check  check the model endpoint and local dashboard\n' >&2
}

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

probe_model() {
  model_payload=""
  if ! model_payload="$(
    curl --noproxy '*' --silent --show-error --fail --max-time 10 "$models_url" 2>/dev/null
  )"; then
    return 2
  fi
  if python3 -c '
import json
import sys

payload = json.load(sys.stdin)
model_ids = [str(item.get("id")) for item in payload.get("data", [])]
raise SystemExit(0 if sys.argv[1] in model_ids else 1)
' "$expected_model" <<<"$model_payload"; then
    return 0
  fi
  return 1
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
prime_rl_path="$prime_rl_root/.venv/bin:${PATH:-}"
env_vars_json="$(python3 -c '
import json
import sys

print(json.dumps({"PATH": sys.argv[1]}))
' "$prime_rl_path")"

mode="${1:-start}"
case "$mode" in
  check)
    if probe_model; then
      printf 'Direct model endpoint is ready: %s -> %s\n' "$models_url" "$expected_model"
    else
      status=$?
      if [[ $status -eq 1 ]]; then
        printf 'Direct model endpoint returned unexpected model(s): %s\n' "$(print_observed_models)" >&2
      else
        printf 'Direct model endpoint is not reachable: %s\n' "$models_url" >&2
      fi
      exit "$status"
    fi
    if ! curl --noproxy '*' --silent --show-error --fail --max-time 5 \
      "$dashboard_url/api/runs" >/dev/null; then
      printf 'Prime-RL dashboard is not reachable: %s\n' "$dashboard_url" >&2
      exit 1
    fi
    printf 'Prime-RL dashboard is ready: %s\n' "$dashboard_url"
    exit 0
    ;;
  start) ;;
  *)
    usage
    exit 2
    ;;
esac

for command_name in bash curl grep prime python3 tee tmux uv; do
  command -v "$command_name" >/dev/null
done

if [[ ! -d "$prime_rl_root" || ! -f "$config_path" ]]; then
  printf 'Prime-RL checkout or config is missing under %s\n' "$project_root" >&2
  exit 1
fi
for executable in torchrun vllm-router; do
  executable_path="$prime_rl_root/.venv/bin/$executable"
  if [[ ! -x "$executable_path" ]]; then
    printf 'Prime-RL dependency is missing: %s. Run: cd %s && uv sync --all-extras\n' \
      "$executable" "$prime_rl_root" >&2
    exit 1
  fi
  if ! "$executable_path" --help >/dev/null 2>&1; then
    printf 'Prime-RL entrypoint is broken: %s. Run: cd %s && uv sync --all-extras --reinstall\n' \
      "$executable" "$prime_rl_root" >&2
    exit 1
  fi
done
if ! (cd "$prime_rl_root" && uv run --no-sync python -c 'import flash_attn' >/dev/null); then
  printf 'Prime-RL dependency is missing: flash_attn. Run: cd %s && uv sync --all-extras\n' \
    "$prime_rl_root" >&2
  exit 1
fi
mkdir -p "$output_root"
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid tmux session name: %s\n' "$session_name" >&2
  exit 2
fi
if [[ ! "$ready_timeout" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PRIME_RL_READY_TIMEOUT must be a positive integer.\n' >&2
  exit 2
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

# A reachable endpoint means another process already owns the managed vLLM
# port. Refuse to launch even when it serves the expected model.
if probe_model; then
  printf 'Refusing to start: %s already serves %s\n' "$models_url" "$expected_model" >&2
  exit 1
else
  preflight_status=$?
  if [[ $preflight_status -eq 1 ]]; then
    printf 'Refusing to start: %s serves unexpected model(s): %s\n' \
      "$models_url" "$(print_observed_models)" >&2
    exit 1
  fi
fi

printf -v train_command \
  'prime env install bird-text2sql --path %q --plain && exec env PATH=%q NO_PROXY=%q no_proxy=%q uv run --no-sync rl @ %q --run.name %q --env-vars %q' \
  "$project_root/environments" "$prime_rl_path" "$NO_PROXY" "$no_proxy" "$config_path" "$run_name" "$env_vars_json"
printf -v train_pipeline '{ %s; } 2>&1 | tee %q' "$train_command" "$startup_log"
printf -v tmux_command \
  'tmux set-option -w -t %q remain-on-exit failed && exec bash -o pipefail -c %q' \
  "$session_name:train" "$train_pipeline"
tmux new-session -d -s "$session_name" -n train -c "$prime_rl_root" "$tmux_command"
printf -v dashboard_command \
  'exec uv run --no-sync dashboard %q --host 127.0.0.1 --port 7788' \
  "$output_root"
tmux new-window -d -t "$session_name:" -n dashboard -c "$prime_rl_root" "$dashboard_command"
printf 'Started tmux session %s (windows: train, dashboard) with NO_PROXY=%s\n' \
  "$session_name" "$NO_PROXY"
printf 'Run name (local file monitor): %s\n' "$run_name"
printf 'Startup log: %s\n' "$startup_log"
printf 'Dashboard: %s\n' "$dashboard_url"

deadline=$((SECONDS + ready_timeout))
while ((SECONDS < deadline)); do
  if ! tmux list-windows -t "$session_name" -F '#{window_name}' 2>/dev/null | grep -Fxq train; then
    printf 'Prime-RL exited before the expected model became ready.\n' >&2
    exit 1
  fi
  if [[ "$(tmux display-message -p -t "$session_name:train" '#{pane_dead}')" == 1 ]]; then
    printf 'Prime-RL exited before the expected model became ready. See %s\n' \
      "$startup_log" >&2
    exit 1
  fi
  if probe_model; then
    printf 'Direct startup check passed: %s -> %s\n' "$models_url" "$expected_model"
    exit 0
  fi
  sleep 2
done

printf 'Timed out after %ss waiting for %s at %s\n' \
  "$ready_timeout" "$expected_model" "$models_url" >&2
if [[ -n "$model_payload" ]]; then
  printf 'Last observed model(s): %s\n' "$(print_observed_models)" >&2
fi
exit 1
