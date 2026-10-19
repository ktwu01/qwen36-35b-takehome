"""Seismic serving lanes from the vLLM logs -> logs/eval_lanes.json (input to scripts/cost_report.py).

    python scripts/eval_lanes.py

One lane per vLLM API-server process: start = its first log line, end = its last log line (a log file
that was reused after a restart holds several processes). Year is 2026 (vLLM prints MM-DD only).
Test logs (vllm_*_test.log) are excluded.
"""

import json
import re
from pathlib import Path

GPU = {"seismic15": "NVIDIA RTX A4500", "seismic17": "NVIDIA RTX A4500",
       "seismic18": "NVIDIA RTX PRO 4000 Blackwell"}
LINE = re.compile(r"\(APIServer pid=(\d+)\) \w+ (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})")


def main():
    lanes = []
    for f in sorted(Path("logs").glob("vllm_*.log")):
        if f.stem.endswith("_test"):
            continue
        parts = f.stem.split("_")  # vllm_<model>[_<host>]; vllm_base.log ran on seismic18
        model, host = parts[1], parts[2] if len(parts) > 2 else "seismic18"
        seg = {}
        for m in LINE.finditer(f.read_text(errors="replace")):
            ts = f"2026-{m[2]}-{m[3]}T{m[4]}"
            s = seg.setdefault(m[1], [ts, ts])
            s[1] = ts
        for pid, (start, end) in sorted(seg.items(), key=lambda kv: kv[1][0]):
            lanes.append({"host": host, "gpu": GPU[host], "model": model, "pid": int(pid),
                          "start": start, "end": end, "log": str(f)})
    Path("logs/eval_lanes.json").write_text(json.dumps(lanes, indent=1))
    for l in lanes:
        print(l["host"], l["model"], l["start"], l["end"])


if __name__ == "__main__":
    main()
