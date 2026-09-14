#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
config="$project_root/configs/prime-rl/eval/arcwise-plat-openrouter-deepseek-r1.toml"
key_file="$project_root/api_key.txt"
output_root="$project_root/outputs/prime-rl"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
mode="${1:-formal}"
http_proxy_value="${http_proxy:-${HTTP_PROXY:-}}"
https_proxy_value="${https_proxy:-${HTTPS_PROXY:-$http_proxy_value}}"
no_proxy_value="${NO_PROXY:-${no_proxy:-}}"

if [[ -z "$http_proxy_value" || -z "$https_proxy_value" ]]; then
  printf 'The local HTTP/HTTPS proxy is not configured in the launcher environment.\n' >&2
  exit 1
fi
for host in localhost 127.0.0.1 127.0.1.1 ::1; do
  case ",$no_proxy_value," in
    *",$host,"*) ;;
    *) no_proxy_value="${no_proxy_value:+$no_proxy_value,}$host" ;;
  esac
done

case "$mode" in
  smoke)
    num_tasks="${BIRD_EVAL_NUM_TASKS_OVERRIDE:-3}"
    max_concurrent=1
    run_kind="smoke"
    verbose_arg="--verbose"
    ;;
  formal)
    num_tasks=498
    max_concurrent=16
    run_kind="formal"
    verbose_arg=""
    ;;
  *)
    printf 'Usage: %s [smoke|formal]\n' "$0" >&2
    exit 2
    ;;
esac
if [[ ! "$num_tasks" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BIRD_EVAL_NUM_TASKS_OVERRIDE must be a positive integer.\n' >&2
  exit 2
fi

for command_name in prime tmux uv; do
  if ! command -v "$command_name" >/dev/null; then
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done
for required in "$prime_rl_root" "$config" "$key_file"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  fi
done
if [[ ! -s "$key_file" ]]; then
  printf 'OpenRouter API key file is empty: %s\n' "$key_file" >&2
  exit 1
fi

batch_id="$(date +%Y%m%d-%H%M%S)"
run_name="bird-deepseek-r1-arcwise-plat-${run_kind}--$batch_id"
session_name="${BIRD_TMUX_SESSION:-$run_name}"
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

# Read the one-time credential only inside the tmux shell. It is neither embedded in
# the command line nor persisted in the resolved evaluation config.
printf -v eval_command \
  'set -o pipefail; export OPENROUTER_API_KEY="$(tr -d '\''\\r\\n'\'' < %q)"; test -n "$OPENROUTER_API_KEY"; export http_proxy=%q https_proxy=%q HTTP_PROXY=%q HTTPS_PROXY=%q no_proxy=%q NO_PROXY=%q; env UV_CACHE_DIR=%q prime env install bird-text2sql --path %q --plain && env UV_CACHE_DIR=%q uv run --no-sync eval @ %q --num-tasks %q --max-concurrent %q --run.name %q %s; status=$?; unset OPENROUTER_API_KEY; exit $status' \
  "$key_file" "$http_proxy_value" "$https_proxy_value" "$http_proxy_value" \
  "$https_proxy_value" "$no_proxy_value" "$no_proxy_value" "$uv_cache_dir" \
  "$project_root/environments" "$uv_cache_dir" "$config" "$num_tasks" \
  "$max_concurrent" "$run_name" "$verbose_arg"

tmux new-session -d -s "$session_name" -n eval -c "$prime_rl_root" \
  "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started %s evaluation in tmux session %s\n' "$run_kind" "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Results: %s/%s\n' "$output_root" "$run_name"
