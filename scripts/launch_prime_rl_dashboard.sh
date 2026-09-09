#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prime_rl_root="$project_root/prime-rl"
output_root="$project_root/outputs/prime-rl"
session_name="${PRIME_RL_DASHBOARD_TMUX_SESSION:-bird-prime-dashboard}"
dashboard_url="${PRIME_RL_DASHBOARD_URL:-http://127.0.0.1:7788}"

case "${1:-start}" in
  check)
    if curl --noproxy '*' --silent --show-error --fail --max-time 5 \
      "$dashboard_url/api/runs" >/dev/null; then
      printf 'Prime-RL dashboard is ready: %s\n' "$dashboard_url"
      exit 0
    fi
    printf 'Prime-RL dashboard is not reachable at %s\n' "$dashboard_url" >&2
    exit 1
    ;;
  start) ;;
  *)
    printf 'Usage: %s [start|check]\n' "$0" >&2
    exit 2
    ;;
esac

for command_name in curl tmux uv; do
  command -v "$command_name" >/dev/null
done
for required_path in "$prime_rl_root"; do
  if [[ ! -d "$required_path" ]]; then
    printf 'Required directory is missing: %s\n' "$required_path" >&2
    exit 1
  fi
done
mkdir -p "$output_root"
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Invalid tmux session name: %s\n' "$session_name" >&2
  exit 2
fi
if curl --noproxy '*' --silent --show-error --fail --max-time 5 \
  "$dashboard_url/api/runs" >/dev/null 2>&1; then
  printf 'Prime-RL dashboard is already ready: %s\n' "$dashboard_url"
  exit 0
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

printf -v tmux_command \
  'exec uv run --no-sync dashboard %q --host 127.0.0.1 --port 7788' \
  "$output_root"
tmux new-session -d -s "$session_name" -c "$prime_rl_root" "$tmux_command"
printf 'Started Prime-RL dashboard in tmux session %s\n' "$session_name"
printf 'Open: %s\n' "$dashboard_url"
