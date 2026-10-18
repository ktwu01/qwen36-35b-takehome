"""Build the result tables, failure lists and uncertainty estimates from the raw runs.

    python scripts/summarize.py --base /var/tmp/$USER --models base swe term sci merged

Writes results/<group>/<benchmark>/<model>/per_task.csv, results/summary.csv,
results/failures/<model>_<benchmark>.csv and analysis/uncertainty.json.
Primary metrics (percent, fixed before training): TB2 = mean success over 5 seeds (all 89 tasks
per seed); SWE-Pro = resolved rate on HARD-51 after patch_replay regrade; HLE = judge accuracy on
all 2,500 questions; ASI-Bench = official mean_percent (scorer-invalid excluded, as asibench
reports) with the all-instances mean (invalid = 0) reported next to it.
Paired bootstrap over tasks (10,000 resamples, seed 0) for merged - base per benchmark and for the
four-benchmark macro average; TB2 uses per-task means over seeds and also reports the seed std.
"""

import argparse
import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takehome import collect  # noqa: E402

GROUP = {"base": "base", "merged": "merged"}
TARGETS = {"swe": {"swe_pro"}, "term": {"tb2"}, "sci": {"hle", "asi"}}


def per_task_means(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["score"])
    return {t: sum(v) / len(v) for t, v in by.items()}


def metric(bench, rows, extra=None):
    if not rows:
        return None
    m = {"n_rows": len(rows), "failure_counts": dict(_count(r["status"] for r in rows))}
    if bench == "tb2":
        seeds = defaultdict(list)
        for r in rows:
            seeds[r["run"]].append(r["score"])
        per_seed = {s: 100 * sum(v) / 89 for s, v in seeds.items()}  # all 89 tasks per seed
        m.update(primary=st.mean(per_seed.values()), seed_std=st.pstdev(per_seed.values()) if len(per_seed) > 1 else 0.0,
                 per_seed=per_seed, n_seeds=len(per_seed), complete=all(len(v) == 89 for v in seeds.values()))
    elif bench == "asi":
        all_mean = 100 * sum(r["score"] for r in rows) / len(rows)
        lv = defaultdict(list)
        for r in rows:
            lv[r["level"]].append(r["score"])
        m.update(primary=extra.get("official_mean_percent"), all_instances_mean=all_mean,
                 per_level={k: 100 * sum(v) / len(v) for k, v in sorted(lv.items())}, **extra)
    else:
        m.update(primary=100 * sum(r["score"] for r in rows) / len(rows))
        if bench == "hle":
            txt = [r for r in rows if not r.get("has_image")]
            m["text_only_subset_accuracy"] = 100 * sum(r["score"] for r in txt) / max(1, len(txt))
    return m


def _count(xs):
    c = defaultdict(int)
    for x in xs:
        c[x] += 1
    return c


def paired_bootstrap(a, b, n=10_000, seed=0):
    """a, b: {task: score in [0,1]} on the same tasks. Returns mean diff (pp) and 95% CI."""
    tasks = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diffs = [b[t] - a[t] for t in tasks]
    boots = sorted(100 * sum(rng.choice(diffs) for _ in tasks) / len(tasks) for _ in range(n))
    return {"diff_pp": 100 * sum(diffs) / len(tasks), "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n)]],
            "n_tasks": len(tasks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="raw run root on the eval host")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--models", nargs="+", default=["base", "swe", "term", "sci", "merged"])
    a = ap.parse_args()
    out, table, tasks = Path(a.results_root), {}, {}
    for model in a.models:
        group = GROUP.get(model, "experts")
        for bench in ("tb2", "swe_pro", "hle", "asi"):
            extra = {}
            try:
                if bench == "tb2":
                    rows = collect.tb2(a.base, model)
                elif bench == "swe_pro":
                    rows = collect.swe_pro(a.base, model)
                elif bench == "hle":
                    rows = collect.hle(out / group, model)
                else:
                    rows, extra = collect.asi(a.base, model)
            except FileNotFoundError:
                rows = []
            if not rows:
                continue
            d = out / group / bench / model
            d.mkdir(parents=True, exist_ok=True)
            keys = sorted({k for r in rows for k in r})
            with open(d / "per_task.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            fails = [r for r in rows if r["score"] < 1.0]
            (out / "failures").mkdir(exist_ok=True)
            with open(out / "failures" / f"{model}_{bench}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(fails)
            table[(model, bench)] = metric(bench, rows, extra)
            tasks[(model, bench)] = per_task_means(rows)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "benchmark", "primary_percent", "seed_std", "n_rows", "failure_counts", "scope"])
        for (model, bench), m in sorted(table.items()):
            scope = "target" if bench in TARGETS.get(model, set()) else ("all" if model in GROUP else "extra")
            w.writerow([model, bench, round(m["primary"], 2) if m["primary"] is not None else "",
                        round(m.get("seed_std", 0), 2), m["n_rows"], json.dumps(m["failure_counts"]), scope])

    unc = {"metrics": {f"{k[0]}/{k[1]}": v for k, v in table.items()}, "paired_bootstrap_merged_minus_base": {}}
    benches = [b for b in ("tb2", "swe_pro", "hle", "asi") if ("base", b) in tasks and ("merged", b) in tasks]
    for bench in benches:
        unc["paired_bootstrap_merged_minus_base"][bench] = paired_bootstrap(tasks[("base", bench)], tasks[("merged", bench)])
    if len(benches) == 4:
        rng = random.Random(1)
        per = {b: (sorted(set(tasks[("base", b)]) & set(tasks[("merged", b)]))) for b in benches}
        boots = []
        for _ in range(10_000):
            s = 0.0
            for b in benches:
                ts = per[b]
                s += 100 * sum(tasks[("merged", b)][t] - tasks[("base", b)][t] for t in (rng.choice(ts) for _ in ts)) / len(ts)
            boots.append(s / 4)
        boots.sort()
        unc["macro_merged_minus_base"] = {
            "diff_pp": sum(unc["paired_bootstrap_merged_minus_base"][b]["diff_pp"] for b in benches) / 4,
            "ci95": [boots[250], boots[9750]]}
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/uncertainty.json").write_text(json.dumps(unc, indent=1, default=str))
    print(open(out / "summary.csv").read())


if __name__ == "__main__":
    main()
