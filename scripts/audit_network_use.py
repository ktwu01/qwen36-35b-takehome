"""Audit SWE-bench Pro agent trajectories for network use (the locked protocol's offline agent phase
is not enforced by the apptainer backend; see README).

    python scripts/audit_network_use.py --base /var/tmp/$USER --models base swe merged

Scans every executed action in mini-swe-agent trajectories (sbp-<model>[-pN]/*/agent/
mini-swe-agent.trajectory.json) for network-capable commands and writes
analysis/network_audit.json: per model, the tasks with any hit and the matching commands.
Heredoc bodies (file contents written with `cat <<EOF`) are dropped before matching, and a bare URL
is not a hit: license headers and docstrings contain URLs without any network access.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NET = re.compile(r"\b(curl|wget|git\s+(fetch|clone|pull|ls-remote|remote\s+add)|pip3?\s+(install|download)|"
                 r"uv\s+(pip|add|sync)|npm\s+(install|i)\b|yarn\s+add|go\s+(get|mod\s+download)|apt(-get)?\s+install|"
                 r"python3?\s+-m\s+pip\s+install|requests\.get|urllib\.request|socket\.connect)", re.I)
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)
ACTION = re.compile(r"```(?:bash|mswea_bash_command)\s*\n(.*?)\n```", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", nargs="+", default=["base", "swe", "merged"])
    a = ap.parse_args()
    jobs = Path(a.base) / "sbp-jobs"
    out = {}
    for model in a.models:
        hits, n_tasks = defaultdict(list), 0
        for traj in sorted(jobs.glob(f"sbp-{model}*/*/agent/mini-swe-agent.trajectory.json")):
            if traj.parts[-4].endswith("-regrade"):
                continue
            n_tasks += 1
            res = json.loads((traj.parents[1] / "result.json").read_text())
            task = Path(res["task_id"]["path"]).name
            for m in json.loads(traj.read_text()).get("messages", []):
                if m.get("role") != "assistant":
                    continue
                for cmd in ACTION.findall(str(m.get("content") or "")):
                    if NET.search(HEREDOC.sub("<<heredoc", cmd)):
                        hits[task].append(cmd.strip()[:300])
        out[model] = {"tasks_scanned": n_tasks, "tasks_with_network_commands": len(hits),
                      "examples": {t: c[:5] for t, c in sorted(hits.items())}}
        print(model, n_tasks, "tasks scanned,", len(hits), "with network-capable commands")
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/network_audit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
