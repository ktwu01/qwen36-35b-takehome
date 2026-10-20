"""Collect evidence for the merged model's failures (input to analysis/failures.md).

    python scripts/failure_cases.py --base /var/tmp/$USER --n 40     # --n 0: every failed task

For each benchmark, every failed task of the merged model gets: exception / exit status, the last
agent turns (truncated), patch size (SWE-Pro), verifier tail, and whether base or the matching expert
solved the same task (flag `merge_regression`). ASI-Bench: score, first failed gate and the agent's
exit status; HLE: finish reason, output tokens, the judge's extracted answer and reasoning. Writes analysis/failure_evidence.jsonl; the
categorization itself (localization, patch, test understanding, merge-induced regression,
environment/tool, timeout, other) is done by reading this evidence and is recorded in failures.md.
"""

import argparse
import json
import random
import re
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
    """Evidence from the scored trial: job_glob names the part the collector took the row from, and
    trials the collector skips (cancelled, endpoint unavailable) are skipped here too."""
    for res in sorted(Path(base).glob(job_glob + "/*/result.json")):
        r = json.loads(res.read_text())
        if Path((r.get("task_id") or {}).get("path", "")).name != task or not r.get("finished_at"):
            continue
        if (r.get("exception_info") or {}).get("exception_type") in ("CancelledError", "InternalServerError"):
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


def asi_evidence(base, task):
    """task = <instance_id>__<level>; per-instance result written by asibench, trajectory found via the
    path the agent printed."""
    inst, lv = task.rsplit("__", 1)
    res = Path(base) / "asi-runs" / "merged" / inst.split("__")[0] / f"{task}.json"
    try:
        r = json.loads(res.read_text())
    except (OSError, ValueError):
        return {}
    gate = next((g for g in r.get("gate_results") or [] if not g.get("passed")), None)
    log = str((r.get("agent_output") or {}).get("log") or "")
    m = re.search(r"trajectories/(\S+?\.traj\.json)", log.replace("\n", ""))
    exit_status, n_turns = None, None
    if m:
        try:
            msgs = json.loads((res.parents[1] / "trajectories" / m.group(1)).read_text()).get("messages", [])
            exit_status = next((x.get("content") for x in msgs if x.get("role") == "exit"), None)
            n_turns = sum(x.get("role") == "assistant" for x in msgs)
        except (OSError, ValueError):
            pass
    return {"score": r.get("final_score"), "agent_exit": exit_status, "assistant_turns": n_turns,
            "execution_s": r.get("execution_time_seconds"),
            "first_failed_gate": (gate or {}).get("message") or str((gate or {}).get("details"))[:300] if gate else None}


_HLE = {}


def hle_evidence(results_root, task):
    d = Path(results_root) / "merged" / "hle" / "merged" / "run1"
    if d not in _HLE:  # read once: {id: record} for predictions and judgments
        _HLE[d] = tuple({x["id"]: x for x in map(json.loads, open(d / f))} for f in ("predictions.jsonl", "judged.jsonl"))
    p, j = (m.get(task, {}) for m in _HLE[d])
    return {"finish_reason": p.get("finish_reason"), "output_tokens": (p.get("usage") or {}).get("completion_tokens"),
            "has_image": p.get("has_image"), "model_answer": j.get("model_answer"),
            "judge_reasoning": (j.get("reasoning") or j.get("judge_error") or "")[:400]}


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
        for r in (fails if a.n == 0 else fails[: a.n]):
            ev = {"benchmark": bench, "task": r["task"], "run": r.get("run"), "status": r["status"],
                  "merge_regression": r["task"] in ref}
            if bench == "tb2":
                ev.update(harbor_evidence(a.base, f"tb2-jobs/{r['part']}", r["task"]))
            elif bench == "swe_pro":
                # agent trajectory from the part whose regrade was scored
                ev.update(harbor_evidence(a.base, f"sbp-jobs/{r['part'].removesuffix('-regrade')}", r["task"]))
                ev["regrade_part"] = r["part"]
            elif bench == "asi":
                ev.update(asi_evidence(a.base, r["task"]))
            elif bench == "hle":
                ev.update(hle_evidence(a.results_root, r["task"]))
            out.append(ev)
    Path("analysis").mkdir(exist_ok=True)
    with open("analysis/failure_evidence.jsonl", "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(len(out), "failure records;", sum(e["merge_regression"] for e in out), "flagged merge_regression")


if __name__ == "__main__":
    main()
