"""Collect per-task results of every model on every benchmark from the raw run directories.

Each collector returns a list of rows {task, run, score (0-1), status} with status one of
  ok            graded normally (score may be 0)
  agent_timeout agent hit its time budget (graded as is: counts in the denominator)
  llm_call_timeout one model call exceeded the harness' 600 s LiteLLM timeout (counts as failure)
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
        if (r.get("exception_info") or {}).get("exception_type") in ("CancelledError", "InternalServerError"):
            # interrupted by the operator (runner stopped) or the model endpoint was unavailable (vLLM
            # restart / serving-window end): not a model or task-environment outcome; the task is
            # re-run in a continuation part. Same rule for every model.
            continue
        # trial dir names truncate the task name (e.g. all teleport tasks share a prefix); use the path
        task = Path((r.get("task_id") or {}).get("path") or res.parent.name.rsplit("__", 1)[0]).name
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type")
        if exc in ENV_ERRORS:
            status = "env_error"
        elif exc == "AgentTimeoutError":
            status = "agent_timeout"
        elif exc == "Timeout":  # one LLM call exceeded LiteLLM's 600 s default (same for every model)
            status = "llm_call_timeout"
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
    d = Path(results_root) / "hle" / model / run  # scripts/run_hle.sh: results/<group>/hle/<model>/run1
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
    # every locked question is in the denominator: one never answered or never judged scores 0
    seen = {r["task"] for r in rows}
    for t in json.load(open(Path(__file__).resolve().parents[2] / "configs/eval/hle.lock.json"))["tasks"]:
        if t["id"] not in seen:
            rows.append({"task": t["id"], "run": run, "score": 0.0, "status": "no_result",
                         "has_image": t["has_image"]})
    return rows


def asi(base, model):
    """Locked instances only (configs/eval/asi_bench.lock.json: 59 tasks x 4 levels): `asibench run
    --tasks all` also runs the dataset's extra test task, which is not part of the evaluation."""
    rep = json.loads((Path(base) / "asi-runs" / model / "local_score_seed31415.json").read_text())
    lock = json.load(open(Path(__file__).resolve().parents[2] / "configs/eval/asi_bench.lock.json"))
    want = {(i, lv) for i in lock["task_ids"] for lv in lock["prompt_levels"]}
    rows, tot, tot_max, n_invalid = [], 0.0, 0.0, 0
    for r in rep["results"]:
        if (r["instance_id"], r["prompt_level"]) not in want:
            continue
        want.discard((r["instance_id"], r["prompt_level"]))
        st = r.get("evaluation_status")
        score, mx = r.get("final_score"), r.get("max_score") or 100.0
        if score is None:
            n_invalid += 1
        else:
            tot, tot_max = tot + score, tot_max + mx
        rows.append({"task": f"{r['instance_id']}__{r['prompt_level']}", "run": "seed31415",
                     "score": (score or 0.0) / mx,
                     "status": "ok" if st == "completed" and score is not None else "no_result",
                     "level": r["prompt_level"], "official_valid": score is not None})
    for i, lv in sorted(want):  # locked instance with no result (never ran): 0, invalid for the official mean
        rows.append({"task": f"{i}__{lv}", "run": "seed31415", "score": 0.0, "status": "no_result",
                     "level": lv, "official_valid": False})
    # asibench's mean_percent (total score / total max over scorer-valid instances), on the locked set
    return rows, {"official_mean_percent": 100 * tot / tot_max if tot_max else None,
                  "scorer_error_count": n_invalid, "not_run_count": len(want), "instance_count": len(rows),
                  "report_mean_percent_incl_extra_task": rep.get("mean_percent")}
