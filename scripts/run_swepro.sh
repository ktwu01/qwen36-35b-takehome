#!/bin/bash
# SWE-bench Pro v2 HARD-51 under the official locked-protocol tooling (SWE-bench_Pro-os@66f9276):
# host-side mini-swe-agent (text-based config; src/takehome/sbp_agent.py adds the locked agents'
# partial-patch capture) generates a patch within the task's 50-min budget,
# then patch_replay re-grades every captured patch on a pristine image (authoritative score).
#   bash scripts/run_swepro.sh <served name> <api base> [n_concurrent]   (env: SBP_TASKS, JOB)
set -euo pipefail
NAME=$1; API=$2; N=${3:-8}
source "$(dirname "$0")/tb2_env.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)
SBP=$BASE/sbp
TASKS=${SBP_TASKS:-$BASE/sbp-hard51-apptainer}
JOB=${JOB:-$NAME}
if [ "${PART:-1}" -gt 1 ]; then
  # continuation part: only tasks without a finished, non-cancelled agent trial in earlier parts
  JOB=$NAME-p$PART
  SUB=$BASE/sbp-parts/$JOB; mkdir -p "$SUB"
  /usr/bin/python3.12 - "$BASE/sbp-jobs" "$NAME" "$TASKS" "$SUB" <<'PY'
import json, os, sys
from pathlib import Path
jobs, name, tasks, sub = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
done = set()
for res in jobs.glob(f"sbp-{name}*/*/result.json"):
    if res.parent.parent.name.endswith("-regrade"):
        continue
    r = json.loads(res.read_text())
    if r.get("finished_at") and (r.get("exception_info") or {}).get("exception_type") not in ("CancelledError", "InternalServerError"):
        done.add(Path((r.get("task_id") or {}).get("path", "")).name)
for t in sorted(tasks.iterdir()):
    if t.name not in done and not (sub / t.name).exists():
        os.symlink(t.resolve(), sub / t.name)
print(f"part: {len(list(sub.iterdir()))} unfinished tasks")
PY
  TASKS=$SUB
fi
CFG=$REPO/configs/eval/swe_pro_jobs/mini_textbased_$NAME.yaml
"$BASE/venvs/harbor/bin/python" "$REPO/scripts/make_swepro_config.py" "$SBP/v2/tooling/configs/mini_textbased.yaml" "$NAME" "$API" "$CFG"
export PYTHONPATH=$SBP/v2/tooling:$REPO/src HOSTED_VLLM_API_KEY=EMPTY MSWEA_COST_TRACKING=ignore_errors
ENV_ARGS=(-e singularity --ek "singularity_image_cache_dir=$BASE/sif-cache" --mounts-json "$TB2_MOUNTS")
"$HARBOR" run -p "$TASKS" -a takehome.sbp_agent:BudgetedHostMiniSwe -m "hosted_vllm/$NAME" --ak "config_file=$CFG" \
  "${ENV_ARGS[@]}" -n "$N" -o "$BASE/sbp-jobs" --job-name "sbp-$JOB" -y
"$HARBOR" run -p "$TASKS" -a patch_replay:PatchReplayAgent -m replay --ak "source_job=$BASE/sbp-jobs/sbp-$JOB" \
  "${ENV_ARGS[@]}" -n "$N" -o "$BASE/sbp-jobs" --job-name "sbp-$JOB-regrade" -y
