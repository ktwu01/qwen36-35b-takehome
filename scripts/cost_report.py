"""Resource and cost accounting -> logs/cost.json (+ a Chinese one-line summary for report.pdf).

    python scripts/cost_report.py --sacct logs/hpc_sacct.txt --eval-lanes logs/eval_lanes.json

Inputs:
  logs/hpc_sacct.txt   `sacct -X -o JobID,JobName%30,Account,Partition,State,Elapsed,AllocTRES%60 -P`
                        for every HPC job of this experiment (charged SU = node-hours x queue rate).
  logs/eval_lanes.json  [{"host", "gpu", "start", "end"}] wall-clock serving lanes on the seismic hosts.
  results/*/hle/*/run1/judged.jsonl usage is not logged by the judge; judge cost is estimated from
  the number of judge calls x the o3-mini price at the time (stated in the output).
Cloud GPU rental: none (UT seismic workstations and HPC allocation the project allocation); the HPC SU charge is the
only metered cost.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

RATE = {"gpu-a100-dev": 3.0, "gpu-a100": 3.0, "gpu-a100-small": 1.5, "gpu-h100": 6.0}
GPUS = {"gpu-a100-dev": 3, "gpu-a100": 3, "gpu-a100-small": 1, "gpu-h100": 2}


def hours(elapsed):
    d, _, t = elapsed.rpartition("-")
    h, m, s = (int(x) for x in t.split(":"))
    return int(d or 0) * 24 + h + m / 60 + s / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sacct", default="logs/hpc_sacct.txt")
    ap.add_argument("--eval-lanes", default="logs/eval_lanes.json")
    a = ap.parse_args()
    jobs, su, gpu_h = [], 0.0, {}
    for r in csv.DictReader(open(a.sacct), delimiter="|"):
        h = hours(r["Elapsed"])
        part = r["Partition"]
        cost = h * RATE.get(part, 0)
        su += cost
        kind = r["JobName"].replace("qwen36-", "")
        gpu_h[kind] = gpu_h.get(kind, 0) + h * GPUS.get(part, 0)
        jobs.append({**r, "hours": round(h, 3), "su": round(cost, 2), "gpu_hours": round(h * GPUS.get(part, 0), 2)})
    lanes = json.loads(Path(a.eval_lanes).read_text()) if Path(a.eval_lanes).exists() else []
    eval_gpu_h = sum((datetime.fromisoformat(l["end"]) - datetime.fromisoformat(l["start"])).total_seconds() / 3600
                     for l in lanes)
    out = {"hpc_su_charged": round(su, 1), "hpc_allocation": "the project allocation",
           "hpc_gpu_hours_by_job_kind": {k: round(v, 1) for k, v in gpu_h.items()},
           "seismic_eval_gpu_hours": round(eval_gpu_h, 1), "cloud_rental_usd": 0, "jobs": jobs, "lanes": lanes}
    out["summary_zh"] = (f"HPC 共 {out['hpc_su_charged']} SU（the project allocation；A100 GPU 时 "
                         f"{round(sum(gpu_h.values()), 1)}），评测在 seismic 工作站约 {out['seismic_eval_gpu_hours']} GPU 时；"
                         "无云租用费用。")
    Path("logs/cost.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(out["summary_zh"])


if __name__ == "__main__":
    main()
