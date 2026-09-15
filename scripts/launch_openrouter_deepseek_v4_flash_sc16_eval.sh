#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
config="$project_root/configs/prime-rl/eval/arcwise-plat-sql-openrouter-deepseek-v4-flash-sc16.toml"
key_file="$project_root/api_key.txt"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
mode="${1:-formal}"
resume_arg=""
http_proxy_value="${http_proxy:-${HTTP_PROXY:-}}"
https_proxy_value="${https_proxy:-${HTTPS_PROXY:-$http_proxy_value}}"
no_proxy_value="${NO_PROXY:-${no_proxy:-}}"

case "$mode" in
  smoke)
    num_tasks="${BIRD_EVAL_NUM_TASKS_OVERRIDE:-2}"
    max_concurrent=4
    run_kind="smoke"
    ;;
  formal)
    num_tasks=498
    max_concurrent=32
    run_kind="formal"
    ;;
  resume)
    if [[ $# -lt 2 ]]; then
      printf 'Resume requires an existing run name.\n' >&2
      exit 2
    fi
    num_tasks=498
    max_concurrent=32
    run_kind="resume"
    resume_arg="--resume"
    ;;
  *)
    printf 'Usage: %s [smoke|formal|resume <run-name>]\n' "$0" >&2
    exit 2
    ;;
esac
[[ "$num_tasks" =~ ^[1-9][0-9]*$ ]] || {
  printf 'BIRD_EVAL_NUM_TASKS_OVERRIDE must be a positive integer.\n' >&2
  exit 2
}
if [[ -z "$http_proxy_value" || -z "$https_proxy_value" ]]; then
  printf 'The local HTTP/HTTPS proxy is not configured.\n' >&2
  exit 1
fi
for host in localhost 127.0.0.1 127.0.1.1 ::1; do
  case ",$no_proxy_value," in
    *",$host,"*) ;;
    *) no_proxy_value="${no_proxy_value:+$no_proxy_value,}$host" ;;
  esac
done

for command_name in prime tmux uv; do
  command -v "$command_name" >/dev/null || {
    printf 'Required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done
for required in "$prime_rl_root" "$config" "$key_file"; do
  [[ -e "$required" ]] || {
    printf 'Required path is missing: %s\n' "$required" >&2
    exit 1
  }
done
[[ -s "$key_file" ]] || {
  printf 'OpenRouter API key file is empty.\n' >&2
  exit 1
}

batch_id="$(date +%Y%m%d-%H%M%S)"
if [[ "$mode" == "resume" ]]; then
  run_name="$2"
  [[ -d "$output_root/$run_name" ]] || {
    printf 'Run directory does not exist: %s\n' "$output_root/$run_name" >&2
    exit 1
  }
  session_name="${BIRD_TMUX_SESSION:-$run_name-resume-$batch_id}"
else
  run_name="bird-deepseek-v4-flash-0731-arcwise-plat-sql-sc16-$run_kind--$batch_id"
  session_name="${BIRD_TMUX_SESSION:-$run_name}"
fi
tmux has-session -t "$session_name" 2>/dev/null && {
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
}
traces="$output_root/$run_name/traces.jsonl"
summary="$output_root/$run_name/sc-summary.json"

# The credential is read only inside the tmux shell and is not embedded in the
# command line, resolved config, trace, or summary.
printf -v eval_command 'set -o pipefail; export OPENROUTER_API_KEY="$(tr -d '\''\r\n'\'' < %q)"; test -n "$OPENROUTER_API_KEY"; export http_proxy=%q https_proxy=%q HTTP_PROXY=%q HTTPS_PROXY=%q no_proxy=%q NO_PROXY=%q; env UV_CACHE_DIR=%q prime env install bird-text2sql --path %q --plain && env UV_CACHE_DIR=%q uv run --no-sync eval @ %q --num-tasks %q --max-concurrent %q --run.name %q %s && env UV_CACHE_DIR=%q uv run %q %q --expected-rollouts 16 --output %q; status=$?; unset OPENROUTER_API_KEY; exit $status' "$key_file" "$http_proxy_value" "$https_proxy_value" "$http_proxy_value" "$https_proxy_value" "$no_proxy_value" "$no_proxy_value" "$uv_cache_dir" "$project_root/environments" "$uv_cache_dir" "$config" "$num_tasks" "$max_concurrent" "$run_name" "$resume_arg" "$uv_cache_dir" "$project_root/scripts/summarize_sc.py" "$traces" "$summary"

tmux new-session -d -s "$session_name" -n eval -c "$prime_rl_root" "bash -lc $(printf %q "$eval_command")"
tmux set-window-option -t "$session_name:eval" remain-on-exit on >/dev/null

printf 'Started DeepSeek V4 Flash SC-16 %s evaluation in tmux session %s\n' "$run_kind" "$session_name"
printf 'Run: %s\n' "$run_name"
printf 'Summary: %s\n' "$summary"
