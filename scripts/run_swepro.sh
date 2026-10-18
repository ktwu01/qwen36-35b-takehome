#!/bin/bash
# SWE-bench Pro v2 HARD-51 under the official locked-protocol tooling (SWE-bench_Pro-os@66f9276):
# host-side mini-swe-agent (text-based config) generates a patch within the task's 50-min budget,
# then patch_replay re-grades every captured patch on a pristine image (authoritative score).
#   bash scripts/run_swepro.sh <served name> <api base> [n_concurrent]
set -euo pipefail
NAME=$1; API=$2; N=${3:-8}
source "$(dirname "$0")/tb2_env.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)
SBP=$BASE/sbp
TASKS=$BASE/sbp-hard51-apptainer
CFG=$REPO/configs/eval/swe_pro_jobs/mini_textbased_$NAME.yaml
"$BASE/venvs/harbor/bin/python" "$REPO/scripts/make_swepro_config.py" "$SBP" "$NAME" "$API" "$CFG"
export PYTHONPATH=$SBP/v2/tooling HOSTED_VLLM_API_KEY=EMPTY MSWEA_COST_TRACKING=ignore_errors
ENV_ARGS=(-e singularity --ek "singularity_image_cache_dir=$BASE/sif-cache" --mounts-json "$TB2_MOUNTS")
"$HARBOR" run -p "$TASKS" -a host_mini_swe:HostMiniSweAgent -m "hosted_vllm/$NAME" --ak "config_file=$CFG" \
  "${ENV_ARGS[@]}" -n "$N" -o "$BASE/sbp-jobs" --job-name "sbp-$NAME" -y
"$HARBOR" run -p "$TASKS" -a patch_replay:PatchReplayAgent -m replay --ak "source_job=$BASE/sbp-jobs/sbp-$NAME" \
  "${ENV_ARGS[@]}" -n "$N" -o "$BASE/sbp-jobs" --job-name "sbp-$NAME-regrade" -y
