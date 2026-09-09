#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^ubuntu@[A-Za-z0-9._-]+$ ]]; then
  printf 'Usage: %s ubuntu@HOST\n' "$0" >&2
  exit 2
fi

remote="$1"
local_project="/home/ubuntu/workspace/bird-text2sql-rl"
remote_project="/home/ubuntu/bird-text2sql-rl"
local_data="/data/ximo/bird-text2sql-rl/data/rl-v1"
remote_data="/data/ximo/bird-text2sql-rl/data/rl-v1"
local_ifeval="/data/ximo/ifeval"
remote_ifeval="/data/ximo/ifeval"
control_path="${XIMO_ML_SSH_CONTROL_PATH:-}"

ssh_options=(-F /dev/null -o BatchMode=yes)
rsync_ssh="ssh -F /dev/null -o BatchMode=yes"
if [[ -n "$control_path" ]]; then
  if [[ ! "$control_path" =~ ^/tmp/[A-Za-z0-9._-]+$ || ! -S "$control_path" ]]; then
    printf 'Invalid SSH control socket: %s\n' "$control_path" >&2
    exit 1
  fi
  ssh_options+=(-S "$control_path")
  rsync_ssh+=" -S $control_path"
fi

test -f "$local_data/manifest.json"
test -f "$local_ifeval/ifeval_input_data.jsonl"
ssh "${ssh_options[@]}" "$remote" \
  'test "$(id -un)" = ubuntu && mkdir -p /home/ubuntu/bird-text2sql-rl /data/ximo/bird-text2sql-rl/data/rl-v1 /data/ximo/ifeval'

rsync -a --checksum --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude 'data/processed/' \
  --exclude 'wandb/' \
  --exclude '/prime-rl/' \
  -e "$rsync_ssh" \
  "$local_project/" "$remote:$remote_project/"

rsync -a --checksum --delete -e "$rsync_ssh" "$local_data/" "$remote:$remote_data/"
rsync -a --checksum --delete --exclude 'hf_cache/' -e "$rsync_ssh" "$local_ifeval/" "$remote:$remote_ifeval/"

local_hashes="$({
  sha256sum "$local_data/manifest.json"
  sha256sum "$local_ifeval/ifeval_input_data.jsonl"
  sha256sum /data/qwen3-4b-instruct-2507/config.json
  sha256sum /data/ximo/sql-training/bird23_train_filtered.jsonl
} | awk '{print $1}')"
remote_hashes="$(ssh "${ssh_options[@]}" "$remote" \
  'sha256sum /data/ximo/bird-text2sql-rl/data/rl-v1/manifest.json /data/ximo/ifeval/ifeval_input_data.jsonl /data/qwen3-4b-instruct-2507/config.json /data/ximo/sql-training/bird23_train_filtered.jsonl' \
  | awk '{print $1}')"

if [[ "$local_hashes" != "$remote_hashes" ]]; then
  printf 'Remote fingerprint mismatch.\n' >&2
  exit 1
fi

ssh "${ssh_options[@]}" "$remote" \
  "awk '\$1 == \"machine\" && (\$2 == \"api.wandb.ai\" || \$2 == \"api.wandb.com\") { found=1 } END { exit(found ? 0 : 1) }' /home/ubuntu/.netrc"

printf 'Project, RL data, IFEval, base model, BIRD source, and W&B credential presence verified.\n'
