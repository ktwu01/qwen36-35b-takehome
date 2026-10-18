"""Remove per-trial apptainer sandbox copies that no running process uses.

    python scripts/cleanup_trials.py [/var/tmp/$USER/sif-cache/trials] [--dry-run]

A trial copy is normally deleted by the patched SingularityEnvironment.stop(); copies leak
when harbor is killed. A copy is removed only if no process's command line references it.
"""

import os
import shutil
import sys
from pathlib import Path

root = Path(next((a for a in sys.argv[1:] if not a.startswith("--")),
                 f"/var/tmp/{os.environ['USER']}/sif-cache/trials"))
dry = "--dry-run" in sys.argv
in_use = set()
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        continue
    for part in cmd.split():
        if str(root) in part:
            in_use.add(Path(part).relative_to(root).parts[0] if Path(part).is_relative_to(root) else part)
removed = 0
for d in sorted(root.iterdir()):
    if d.name in in_use:
        continue
    removed += 1
    if not dry:
        shutil.rmtree(d, ignore_errors=True)
print(f"{'would remove' if dry else 'removed'} {removed}, kept {len(in_use)} in use")
