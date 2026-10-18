#!/bin/bash
# Validate the apptainer environment on SWE-bench Pro v2 HARD-51 with the reference patch
# (oracle) and the empty patch (nop). Official gate: oracle 51/51, nop 0/51.
#   bash scripts/run_swepro_oracle.sh [n_concurrent]
set -euo pipefail
source "$(dirname "$0")/tb2_env.sh"
SBP_TASKS=${SBP_TASKS:-$BASE/sbp-hard51}
for agent in oracle nop; do
  "$HARBOR" run -p "$SBP_TASKS" -a "$agent" -e singularity --ek "singularity_image_cache_dir=$BASE/sif-cache" \
    --mounts-json "$TB2_MOUNTS" -n "${1:-8}" -o "$BASE/sbp-jobs" --job-name "sbp-hard51-$agent" -y
done
