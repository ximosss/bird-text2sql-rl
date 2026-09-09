#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
uv_cache_dir="${UV_CACHE_DIR:-/tmp/bird-text2sql-rl-uv-cache}"
local_api_key="${LOCAL_API_KEY:-local}"
best_step="${BIRD_EVAL_BEST_STEP:-80}"
final_step="${BIRD_EVAL_FINAL_STEP:-120}"
base_model_id="bird-text2sql-base"
best_model_id="bird-text2sql-rl-step$best_step"
final_model_id="bird-text2sql-rl-step$final_step"
session_name="${BIRD_TMUX_SESSION:-bird-rl-v1}"

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

export NO_PROXY="$(merge_no_proxy)"
export no_proxy="$NO_PROXY"

usage() {
  printf 'Usage: %s start <direct|generalization|full-dev|v2|sft-base-smoke|sft-base|all>\n' "$0" >&2
  printf '       %s check <direct|generalization|full-dev|v2|sft-base-smoke|sft-base|all>\n' "$0" >&2
}

window_for_suite() {
  local suite="$1"
  printf '%s' "eval-$suite"
}

validate_suite() {
  case "$1" in
    direct|generalization|full-dev|v2|sft-base-smoke|sft-base|all) ;;
    *) usage; exit 2 ;;
  esac
}

run_one() {
  local config_name="$1"
  local model_id="$2"
  local run_name="$3"
  shift 3
  printf '\nStarting eval run: %s\n' "$run_name"
  uv run --no-sync eval @ "$project_root/configs/prime-rl/eval/$config_name.toml" \
    --model "$model_id" \
    --run.name "$run_name" \
    "$@"
}

run_suite() {
  local suite="$1"
  local batch_id="$2"

  cd "$prime_rl_root"
  env UV_CACHE_DIR="$uv_cache_dir" prime env install bird-text2sql --path "$project_root/environments" --plain

  if [[ "$suite" == direct || "$suite" == all ]]; then
    run_one direct-id "$base_model_id" "direct-id-base--$batch_id"
    run_one direct-id "$best_model_id" "direct-id-step$best_step--$batch_id"
    run_one direct-id "$final_model_id" "direct-id-step$final_step--$batch_id"
    run_one direct-ood "$base_model_id" "direct-ood-base--$batch_id"
    run_one direct-ood "$best_model_id" "direct-ood-step$best_step--$batch_id"
    run_one direct-ood "$final_model_id" "direct-ood-step$final_step--$batch_id"
  fi
  if [[ "$suite" == full-dev || "$suite" == all ]]; then
    run_one full-dev "$base_model_id" "full-dev-base--$batch_id"
    run_one full-dev "$best_model_id" "full-dev-step$best_step--$batch_id"
    run_one full-dev "$final_model_id" "full-dev-step$final_step--$batch_id"
  fi
  if [[ "$suite" == generalization || "$suite" == all ]]; then
    for benchmark in arcwise-plat arcwise-plat-sql bird-mini-dev bird-full-dev; do
      run_one "$benchmark" "$base_model_id" "$benchmark-base--$batch_id"
      run_one "$benchmark" "$best_model_id" "$benchmark-step$best_step--$batch_id"
      run_one "$benchmark" "$final_model_id" "$benchmark-step$final_step--$batch_id"
    done
  fi
  if [[ "$suite" == v2 ]]; then
    run_one platinum-v2 "$base_model_id" "v2-platinum-base--$batch_id"
    run_one platinum-v2 "$best_model_id" "v2-platinum-step$best_step--$batch_id"
    run_one platinum-v2 "$final_model_id" "v2-platinum-step$final_step--$batch_id"
    run_one full-dev-v2 "$base_model_id" "v2-full-dev-base--$batch_id"
    run_one full-dev-v2 "$best_model_id" "v2-full-dev-step$best_step--$batch_id"
    run_one full-dev-v2 "$final_model_id" "v2-full-dev-step$final_step--$batch_id"
  fi
  if [[ "$suite" == sft-base ]]; then
    run_one sft-train-base-v3 "$base_model_id" "sft-train-base-v3--$batch_id"
  fi
  if [[ "$suite" == sft-base-smoke ]]; then
    run_one sft-train-base-v3 "$base_model_id" "sft-train-base-smoke-v3--$batch_id" \
      --env.taskset.limit 8
  fi
}

mode="${1:-}"
suite="${2:-}"
if [[ "$mode" == run-suite ]]; then
  validate_suite "$suite"
  run_suite "$suite" "${3:?missing batch id}"
  exit 0
fi
validate_suite "$suite"

window_name="$(window_for_suite "$suite")"
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ || ! "$window_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid tmux target: %s:%s\n' "$session_name" "$window_name" >&2
  exit 2
fi

case "$mode" in
  check)
    "$project_root/scripts/launch_eval_server.sh" check
    if pane_state="$(tmux list-panes -t "$session_name:$window_name" -F '#{pane_dead} #{pane_dead_status}' 2>/dev/null)"; then
      read -r pane_dead pane_status <<<"$pane_state"
      if [[ "$pane_dead" == 0 ]]; then
        printf 'Eval suite is running in tmux window: %s:%s\n' "$session_name" "$window_name"
      else
        printf 'Eval suite has exited in tmux window %s:%s (status %s)\n' "$session_name" "$window_name" "$pane_status"
      fi
    else
      printf 'Eval suite has not been started: %s:%s\n' "$session_name" "$window_name"
    fi
    printf 'Runs are stored under: %s\n' "$output_root"
    ;;
  start)
    for command_name in prime rg tmux uv; do
      command -v "$command_name" >/dev/null
    done
    for required_path in "$prime_rl_root"; do
      if [[ ! -d "$required_path" ]]; then
        printf 'Required directory is missing: %s\n' "$required_path" >&2
        exit 1
      fi
    done
    mkdir -p "$output_root" "$uv_cache_dir"
    "$project_root/scripts/launch_eval_server.sh" check
    if tmux list-windows -t "$session_name" -F '#{window_name}' 2>/dev/null | rg -Fxq "$window_name"; then
      printf 'tmux window already exists: %s:%s\n' "$session_name" "$window_name" >&2
      exit 1
    fi
    batch_id="$(date +%Y%m%d-%H%M%S)"
    printf -v tmux_command \
      'exec env LOCAL_API_KEY=%q NO_PROXY=%q no_proxy=%q BIRD_EVAL_BEST_STEP=%q BIRD_EVAL_FINAL_STEP=%q %q run-suite %q %q' \
      "$local_api_key" "$NO_PROXY" "$no_proxy" "$best_step" "$final_step" "$project_root/scripts/launch_prime_rl_eval.sh" "$suite" "$batch_id"
    tmux new-window -d -t "$session_name:" -n "$window_name" -c "$project_root" "$tmux_command"
    tmux set-window-option -t "$session_name:$window_name" remain-on-exit on >/dev/null
    printf 'Started %s evals in tmux window %s:%s\n' "$suite" "$session_name" "$window_name"
    printf 'Run batch: %s\n' "$batch_id"
    ;;
  *)
    usage
    exit 2
    ;;
esac
