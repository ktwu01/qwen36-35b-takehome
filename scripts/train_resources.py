"""Per-expert training resources over all resumed sessions -> logs/train-<expert>/resources.csv.

    python scripts/train_resources.py

src/takehome/sft.py writes outputs/<expert>/resources.csv at the end of the run, but its wall time
covers only the last session (training resumed across 2 h HPC dev jobs), so tokens_per_s is too high.
Here wall time is the sum over sessions of the last `elapsed_s` before each reset in metrics.jsonl
(includes periodic dev evaluation); tokens, GPU and adapter size are taken from the run's file.
"""

import csv
import json
from pathlib import Path

for e in ("swe", "term", "sci", "union"):
    rows = [json.loads(l) for l in open(f"logs/train-{e}/metrics.jsonl")]
    sessions, prev = [], None
    for r in rows:
        if prev is not None and r["elapsed_s"] < prev:
            sessions.append(prev)
        prev = r["elapsed_s"]
    sessions.append(prev)
    run = next(csv.DictReader(open(f"outputs/{e}/resources.csv")))
    wall = sum(sessions)
    out = {"expert": e, "gpu": run["gpu"], "n_gpu": run["n_gpu"], "sessions": len(sessions),
           "wall_h": round(wall / 3600, 2), "gpu_hours": round(wall / 3600 * int(run["n_gpu"]), 2),
           "steps": max(r["step"] for r in rows), "peak_mem_gb": max(r.get("peak_mem_gb", 0) for r in rows),
           "train_tokens": run["train_tokens"], "tokens_per_s": round(int(run["train_tokens"]) / wall, 1),
           "adapter_mb": run["adapter_mb"]}
    with open(f"logs/train-{e}/resources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out))
        w.writeheader()
        w.writerow(out)
    print(out)
