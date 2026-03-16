#!/usr/bin/env bash

# PPO-Lagrangian 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-ppolag}"
export GPU="${GPU:-4}"

exec "${SCRIPT_DIR}/run_example.sh" "$@"

