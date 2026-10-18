#!/bin/bash
# Run the reference solutions (oracle agent) on all 89 TB2 tasks to validate the
# apptainer environment. Tasks the oracle fails here are environment failures.
#   bash scripts/run_tb2_oracle.sh [n_concurrent]
set -euo pipefail
source "$(dirname "$0")/tb2_env.sh"
"$HARBOR" run -p "$TB2_TASKS" -a oracle "${TB2_ENV_ARGS[@]}" -n "${1:-16}" \
  -o "$BASE/tb2-jobs" --job-name oracle-full -y
