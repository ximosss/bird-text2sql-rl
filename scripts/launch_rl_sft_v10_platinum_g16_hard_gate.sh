#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BIRD_RL_CONFIG="$project_root/configs/prime-rl/rl-sft-v10-platinum-g16-hard-gate-v3.toml"
export BIRD_RL_RUN_PREFIX="bird-rl-sft-v10-plat-full-g16-hard-gate-v3"
export BIRD_RL_DEFAULT_STEPS=20

exec "$project_root/scripts/launch_rl_sft_v10_platinum_g16_format.sh" "$@"
