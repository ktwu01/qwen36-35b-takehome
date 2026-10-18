"""Copy SWE-bench Pro v2 HARD-51 tasks for the apptainer backend.

    python scripts/make_sbp_apptainer_tasks.py /var/tmp/$USER/sbp-hard51 /var/tmp/$USER/sbp-hard51-apptainer

The only change: [agent] network_mode "no-network" -> "public". Harbor's singularity backend
cannot isolate the network (its exec server is reached over host localhost) and rejects
no-network tasks. Every other byte is identical; the diff is written to
configs/eval/swe_pro_task_modifications.json. Trajectories are audited for network use
(scripts/audit_network_use.py) since the locked protocol's offline agent phase is not enforced.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
changes = []
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=True)
for toml in sorted(dst.glob("*/task.toml")):
    text = toml.read_text()
    lines = text.splitlines(keepends=True)
    section, out = None, []
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s
        if section == "[agent]" and s.replace(" ", "") == 'network_mode="no-network"':
            line = line.replace('"no-network"', '"public"')
        out.append(line)
    new = "".join(out)
    assert new.count('"no-network"') == 0, toml
    toml.write_text(new)
    changes.append({"task": toml.parent.name, "sha256_before": hashlib.sha256(text.encode()).hexdigest(),
                    "sha256_after": hashlib.sha256(new.encode()).hexdigest(),
                    "change": '[agent] network_mode "no-network" -> "public"'})
Path("configs/eval").mkdir(parents=True, exist_ok=True)
Path("configs/eval/swe_pro_task_modifications.json").write_text(json.dumps(changes, indent=1))
print(f"{len(changes)} tasks written to {dst}")
