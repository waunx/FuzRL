#!/usr/bin/env bash

# CVaR PPO-Lagrangian

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-cvarppolag}"
export GPU="${GPU:-2}"

exec "${SCRIPT_DIR}/run_example.sh" "$@"

