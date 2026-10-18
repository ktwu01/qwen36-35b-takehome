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
        task = res.parent.name.rsplit("__", 1)[0]
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
        for t in harbor_trials(Path(base) / "tb2-jobs" / f"tb2-{model}-s{s}"):
            rows.append({**t, "run": f"seed{s}"})
    return rows


def swe_pro(base, model):
    """Authoritative score: patch_replay regrade on pristine images."""
    return [{**t, "run": "regrade"} for t in harbor_trials(Path(base) / "sbp-jobs" / f"sbp-{model}-regrade")]


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
