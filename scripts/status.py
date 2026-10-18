"""One-screen status of all evaluation lanes (run on the eval host).

    python3 scripts/status.py
"""

import collections
import glob
import json
import os
import shutil
from pathlib import Path

B = Path(f"/var/tmp/{os.environ['USER']}")


def harbor(pattern):
    c, tasks = collections.Counter(), set()
    for f in glob.glob(str(B / pattern / "*/result.json")):
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        e = (r.get("exception_info") or {}).get("exception_type")
        if not r.get("finished_at") or e == "CancelledError":
            continue
        tasks.add(Path((r.get("task_id") or {}).get("path", f)).name)
        v = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        c["pass" if v == 1.0 else (e or "fail")] += 1
    return len(tasks), dict(c)


for m in ("base", "term", "merged"):
    for s in range(1, 6):
        n, c = harbor(f"tb2-jobs/tb2-{m}-s{s}*")
        if n:
            print(f"TB2 {m:7s} seed{s}: {n:3d}/89 {c}")
for m in ("base", "swe", "merged"):
    n, c = harbor(f"sbp-jobs/sbp-{m}")
    nr, cr = harbor(f"sbp-jobs/sbp-{m}-regrade")
    if n or nr:
        print(f"SWE {m:7s}: agent {n:2d}/51 {c} | regrade {nr} {cr}")
for m in ("base", "sci", "merged"):
    rs = [json.load(open(f)) for f in glob.glob(str(B / "asi-runs" / m / "*/*__b[1-4].json"))]
    if rs:
        print(f"ASI {m:7s}: {len(rs):3d}/236 {dict(collections.Counter(r.get('status') for r in rs))}")
print("HLE access:", "GRANTED " + (B / "HLE_GRANTED").read_text().strip() if (B / "HLE_GRANTED").exists() else "not yet")
du = shutil.disk_usage(B)
print(f"disk free: {du.free / 2**30:.0f} GiB")
