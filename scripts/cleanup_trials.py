"""Remove per-trial apptainer sandbox copies of trials that are finished or discarded.

    python scripts/cleanup_trials.py [--dry-run]

A trial copy (<sif-cache>/trials/<trial_name>__<role>) is normally deleted by the patched
SingularityEnvironment.stop(); copies leak when harbor is killed. Process command lines are
not a reliable liveness signal (the apptainer runtime rewrites its argv), so a copy is removed
only if its trial's result.json records finished_at, or the trial sits in a DISCARDED-* job.
"""

import json
import os
import shutil
import sys
from pathlib import Path

base = Path(f"/var/tmp/{os.environ['USER']}")
root = base / "sif-cache/trials"
dry = "--dry-run" in sys.argv
status = {}
for jobs in ("tb2-jobs", "sbp-jobs"):
    for res in (base / jobs).glob("*/*/result.json"):
        trial, job = res.parent.name, res.parent.parent.name
        try:
            done = bool(json.loads(res.read_text()).get("finished_at"))
        except (OSError, ValueError):
            done = False
        status[trial] = done or job.startswith("DISCARDED-")
removed = kept = 0
for d in sorted(root.iterdir()):
    trial = d.name.rsplit("__", 1)[0]
    if status.get(trial):
        removed += 1
        if not dry:
            shutil.rmtree(d, ignore_errors=True)
    else:
        kept += 1
print(f"{'would remove' if dry else 'removed'} {removed}, kept {kept} (running or unknown)")
