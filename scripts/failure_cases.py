"""Collect evidence for the merged model's failures (input to analysis/failures.md).

    python scripts/failure_cases.py --base /var/tmp/$USER --n 40

For each benchmark, every failed task of the merged model gets: exception / exit status, the last
agent turns (truncated), patch size (SWE-Pro), verifier tail, and whether base or the matching expert
solved the same task (flag `merge_regression`). Writes analysis/failure_evidence.jsonl; the
categorization itself (localization, patch, test understanding, merge-induced regression,
environment/tool, timeout, other) is done by reading this evidence and is recorded in failures.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takehome import collect  # noqa: E402

EXPERT = {"tb2": "term", "swe_pro": "swe", "hle": "sci", "asi": "sci"}


def solved(rows):
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r["score"])
    return {t for t, v in by.items() if max(v) >= 1.0}


def tail(path, n=1500):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError:
        return ""


def harbor_evidence(base, job_glob, task):
    for res in Path(base).glob(job_glob + "/*/result.json"):
        r = json.loads(res.read_text())
        if Path((r.get("task_id") or {}).get("path", "")).name != task:
            continue
        d = res.parent
        traj = d / "agent" / "mini-swe-agent.trajectory.json"
        last = ""
        if traj.exists():
            msgs = json.loads(traj.read_text()).get("messages", [])
            last = "\n---\n".join(str(m.get("content"))[-600:] for m in msgs[-3:])
        elif (d / "agent" / "trajectory.json").exists():
            steps = json.loads((d / "agent" / "trajectory.json").read_text()).get("steps", [])
            last = "\n---\n".join(str(s.get("message"))[-600:] for s in steps[-3:])
        patch = d / "agent" / "model.patch"
        return {"trial": d.name, "exception": (r.get("exception_info") or {}).get("exception_type"),
                "agent_exit": tail(d / "agent" / "result.json", 300), "last_turns": last,
                "patch_bytes": patch.stat().st_size if patch.exists() else None,
                "verifier_tail": tail(d / "verifier" / "test-stdout.txt", 1200)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(0)
    out = []
    getters = {"tb2": lambda m: collect.tb2(a.base, m), "swe_pro": lambda m: collect.swe_pro(a.base, m),
               "asi": lambda m: collect.asi(a.base, m)[0],
               "hle": lambda m: collect.hle(Path(a.results_root) / ("merged" if m == "merged" else
                                                                   "base" if m == "base" else "experts"), m)}
    for bench, get in getters.items():
        try:
            merged = get("merged")
        except FileNotFoundError:
            continue
        ref = set()
        for m in ("base", EXPERT[bench]):
            try:
                ref |= solved(get(m))
            except FileNotFoundError:
                pass
        fails = [r for r in merged if r["score"] < 1.0]
        rng.shuffle(fails)
        for r in fails[: a.n]:
            ev = {"benchmark": bench, "task": r["task"], "run": r.get("run"), "status": r["status"],
                  "merge_regression": r["task"] in ref}
            if bench == "tb2":
                ev.update(harbor_evidence(a.base, f"tb2-jobs/tb2-merged-s{str(r['run'])[-1]}*", r["task"]))
            elif bench == "swe_pro":
                ev.update(harbor_evidence(a.base, "sbp-jobs/sbp-merged*", r["task"]))
            out.append(ev)
    Path("analysis").mkdir(exist_ok=True)
    with open("analysis/failure_evidence.jsonl", "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(len(out), "failure records;", sum(e["merge_regression"] for e in out), "flagged merge_regression")


if __name__ == "__main__":
    main()
