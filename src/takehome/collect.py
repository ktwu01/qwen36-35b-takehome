"""Collect per-task results of every model on every benchmark from the raw run directories.

Each collector returns a list of rows {task, run, score (0-1), status} with status one of
  ok            graded normally (score may be 0)
  agent_timeout agent hit its time budget (graded as is: counts in the denominator)
  env_error     environment / infrastructure failure (counted as failure, also reported apart)
  no_result     generation failed or judge failed (counted as failure)
Failures are never dropped from denominators.
"""

import json
from pathlib import Path

ENV_ERRORS = {"RuntimeError", "ConnectError", "VerifierTimeoutError", "EnvironmentStartTimeoutError",
              "MemoryLimitExceededError", "AgentSetupTimeoutError", "InternalServerError"}


def harbor_trials(job_dir):
    for res in sorted(Path(job_dir).glob("*/result.json")):
        try:
            r = json.loads(res.read_text())
        except ValueError:
            continue
        if not r.get("finished_at"):
            continue
        if (r.get("exception_info") or {}).get("exception_type") == "CancelledError":
            continue  # interrupted by the operator (runner stopped); re-run in a continuation part
        # trial dir names truncate the task name (e.g. all teleport tasks share a prefix); use the path
        task = Path((r.get("task_id") or {}).get("path") or res.parent.name.rsplit("__", 1)[0]).name
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type")
        if exc in ENV_ERRORS:
            status = "env_error"
        elif exc == "AgentTimeoutError":
            status = "agent_timeout"
        elif reward is None:
            status = "env_error" if exc else "no_result"
        else:
            status = "ok"
        a = r.get("agent_result") or {}
        yield {"task": task, "score": float(reward or 0.0), "status": status, "exception": exc,
               "started_at": r.get("started_at"), "finished_at": r.get("finished_at"),
               "input_tokens": a.get("n_input_tokens"), "output_tokens": a.get("n_output_tokens")}


def tb2(base, model, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for s in seeds:
        seen = set()
        # a seed may be split into continuation parts (tb2-<model>-s<seed>-p2, ...); one row per task
        for job in sorted((Path(base) / "tb2-jobs").glob(f"tb2-{model}-s{s}"), key=str) + \
                sorted((Path(base) / "tb2-jobs").glob(f"tb2-{model}-s{s}-p*"), key=str):
            for t in harbor_trials(job):
                if t["task"] not in seen:
                    seen.add(t["task"])
                    rows.append({**t, "run": f"seed{s}", "part": job.name})
    return rows


def swe_pro(base, model):
    """Authoritative score: patch_replay regrade on pristine images."""
    rows, seen = [], set()
    jobs = Path(base) / "sbp-jobs"
    for job in [jobs / f"sbp-{model}-regrade"] + sorted(jobs.glob(f"sbp-{model}-p*-regrade")):
        # only tasks whose agent trial finished in this part (an interrupted part's regrade would
        # replay cancelled tasks with an empty patch)
        agent_done = {t["task"] for t in harbor_trials(jobs / job.name.removesuffix("-regrade"))}
        for t in harbor_trials(job):
            if t["task"] in agent_done and t["task"] not in seen:
                seen.add(t["task"])
                rows.append({**t, "run": "regrade", "part": job.name})
    return rows


def hle(results_root, model, run="run1"):
    d = Path(results_root) / model / "hle" / run
    preds = {r["id"]: r for r in map(json.loads, open(d / "predictions.jsonl"))}
    rows = []
    for r in map(json.loads, open(d / "judged.jsonl")):
        p = preds.get(r["id"], {})
        ok_gen = bool(p.get("response"))
        status = "ok" if ok_gen and r.get("correct") in ("yes", "no") else "no_result"
        usage = p.get("usage") or {}
        rows.append({"task": r["id"], "run": run, "score": 1.0 if r.get("correct") == "yes" else 0.0,
                     "status": status, "has_image": p.get("has_image"), "finish_reason": p.get("finish_reason"),
                     "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                     "latency_s": p.get("latency_s")})
    return rows


def asi(base, model):
    rep = json.loads((Path(base) / "asi-runs" / model / "local_score_seed31415.json").read_text())
    rows = []
    for r in rep["results"]:
        st = r.get("evaluation_status")
        score = r.get("final_score")
        rows.append({"task": f"{r['instance_id']}__{r['prompt_level']}", "run": "seed31415",
                     "score": (score or 0.0) / (r.get("max_possible_score") or 100.0),
                     "status": "ok" if st == "completed" and score is not None else "no_result",
                     "level": r["prompt_level"], "official_valid": score is not None})
    return rows, {"official_mean_percent": rep.get("mean_percent"), "scorer_error_count": rep.get("scorer_error_count"),
                  "instance_count": rep.get("instance_count")}
