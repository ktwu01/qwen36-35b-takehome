"""Freeze the four benchmarks before any training.

Writes configs/eval/<bench>.lock.json with source URL, pinned revision, split and
the complete task-ID list. Only identifiers and metadata are stored: no answers,
reference patches or tests. Run once on a machine with HF access to cais/hle.

    python scripts/lock_benchmarks.py [--only swe_pro hle asi tb2]
"""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

OUT = Path("configs/eval")

SWE_PRO = {"repo": "ScaleAI/SWE-bench_Pro", "config": "hard", "split": "test",
           "revision": "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"}
HLE = {"repo": "cais/hle", "split": "test",
       "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"}
ASI = {"repo": "Apexintelligence-AI/ASI-Bench-seed31415",
       "revision": "0fa14219cafdbab634d8b3cfbce238a8735a214f",
       "package": "asibench==0.2.0", "code": "https://github.com/apexin-ai/ASI-Bench"}
TB2 = {"dataset": "terminal-bench@2.0", "harbor": "harbor==0.24.0",
       "code": "https://github.com/harbor-framework/terminal-bench"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    payload["locked_at"] = now()
    payload["n_tasks"] = len(payload["task_ids"])
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    path = OUT / f"{name}.lock.json"
    path.write_text(text + "\n")
    print(f"{path}: {payload['n_tasks']} tasks sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")


def lock_swe_pro():
    from datasets import load_dataset
    ds = load_dataset(SWE_PRO["repo"], SWE_PRO["config"], split=SWE_PRO["split"],
                      revision=SWE_PRO["revision"])
    assert all(ds["hard"]) and set(ds["version"]) == {"2.0.0"}
    tasks = [{"instance_id": r["instance_id"], "repo": r["repo"], "language": r["repo_language"],
              "docker_image": r["dockerhub_tag"] or r["docker_image"]} for r in ds]
    write("swe_pro_v2_hard", {
        "benchmark": "SWE-bench Pro v2-hard",
        "source": f"https://huggingface.co/datasets/{SWE_PRO['repo']}",
        **SWE_PRO,
        "hard_definition": "dataset config `hard` (eval.yaml: 51-task HARD subset of SWE-Bench Pro V2), every row has hard=True, version=2.0.0",
        "metric": "resolved rate (official SWE-bench_Pro-os evaluator)",
        "task_ids": [t["instance_id"] for t in tasks], "tasks": tasks,
        "excluded_repos_for_training": sorted({t["repo"] for t in tasks}),
    })


def lock_hle():
    from datasets import load_dataset
    ds = load_dataset(HLE["repo"], split=HLE["split"], revision=HLE["revision"])
    ds = ds.remove_columns([c for c in ds.column_names if c in ("answer", "rationale", "image", "rationale_image")])
    rows = [{"id": r["id"], "has_image": bool(r.get("image_preview") or r.get("image")) if "image_preview" in r else None,
             "category": r.get("category"), "answer_type": r.get("answer_type")} for r in ds]
    write("hle", {
        "benchmark": "Humanity's Last Exam",
        "source": f"https://huggingface.co/datasets/{HLE['repo']}", **HLE,
        "metric": "accuracy (official judge prompt, centerforaisafety/hle)",
        "task_ids": [r["id"] for r in rows], "tasks": rows,
    })


def lock_asi():
    path = snapshot_download(ASI["repo"], repo_type="dataset", revision=ASI["revision"],
                             allow_patterns=["metadata.json", "tasks/*/instance_meta.json", "README.md"])
    tasks = sorted(p.name for p in Path(path, "tasks").iterdir())
    write("asi_bench", {
        "benchmark": "ASI-Bench", "source": f"https://huggingface.co/datasets/{ASI['repo']}", **ASI,
        "prompt_levels": ["b1", "b2", "b3", "b4"],
        "metadata": json.loads(Path(path, "metadata.json").read_text()),
        "task_ids": tasks,
    })


def lock_tb2(harbor_bin):
    out = subprocess.run([harbor_bin, "datasets", "download", TB2["dataset"], "-o", "data/bench/tb2"],
                         capture_output=True, text=True)
    print(out.stdout[-2000:], out.stderr[-2000:])
    root = Path("data/bench/tb2")
    tasks = sorted({p.parent.name for p in root.rglob("task.toml")})
    write("terminal_bench_2", {
        "benchmark": "Terminal-Bench 2.0", **TB2,
        "metric": "task success rate, mean and std over 5 runs",
        "task_ids": tasks,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["swe_pro", "hle", "asi", "tb2"])
    ap.add_argument("--harbor-bin", default=str(Path.home() / "venvs/harbor/bin/harbor"))
    a = ap.parse_args()
    print("hf user:", HfApi().whoami()["name"])
    for b in a.only:
        {"swe_pro": lock_swe_pro, "hle": lock_hle, "asi": lock_asi,
         "tb2": lambda: lock_tb2(a.harbor_bin)}[b]()
