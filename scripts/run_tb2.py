"""Terminal-Bench 2.0 with Harbor + Terminus-2 against a served model, 5 fixed seeds.

    python scripts/run_tb2.py --name base --api http://localhost:18000/v1 --seeds 1 2 3 4 5 -n 12

Fixed for every evaluated model (configs/eval/terminal_bench_2.yaml mirrors these):
agent timeout 3 h (override_timeout_sec=10800), 32 CPU / 48 GB per task, temperature 1.0,
top_p 0.95, top_k 20, max_tokens 81920, context 262144, thinking on, interleaved thinking off.
The seed is passed per request to vLLM. Each seed is one Harbor job under
<jobs>/tb2-<name>-s<seed>; re-running resumes unfinished trials.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

BASE = Path(os.environ.get("BASE", f"/var/tmp/{os.environ['USER']}"))
REPO = Path(__file__).resolve().parents[1]


def job_config(name, api, seed, n, jobs_dir):
    mounts = [{"type": "bind", "source": str(BASE / "harbor-rt"), "target": "/opt/harbor-rt", "read_only": True},
              {"type": "bind", "source": "/etc/resolv.conf", "target": "/etc/resolv.conf", "read_only": True},
              {"type": "bind", "source": str(BASE / "harbor-rt/hosts"), "target": "/etc/hosts", "read_only": True}]
    return {
        "job_name": f"tb2-{name}-s{seed}",
        "jobs_dir": str(jobs_dir),
        "n_concurrent_trials": n,
        "n_attempts": 1,
        "environment": {"type": "singularity", "override_cpus": 32, "override_memory_mb": 49152,
                        "mounts": mounts, "kwargs": {"singularity_image_cache_dir": str(BASE / "sif-cache")}},
        "agents": [{
            "name": "terminus-2",
            "model_name": f"hosted_vllm/{name}",
            "override_timeout_sec": 10800,
            "kwargs": {
                "api_base": api,
                "temperature": 1.0,
                "interleaved_thinking": False,
                "model_info": {"max_input_tokens": 262144, "max_output_tokens": 81920, "max_tokens": 81920,
                               "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                "llm_call_kwargs": {"top_p": 0.95, "max_tokens": 81920, "seed": seed,
                                    "extra_body": {"top_k": 20}},
            },
        }],
        "datasets": [{"path": str(REPO / "data/bench/tb2/terminal-bench")}],
    }


def finished_tasks(jobs_dir, name, seed):
    done = set()
    for res in jobs_dir.glob(f"tb2-{name}-s{seed}*/*/result.json"):
        try:
            if json.loads(res.read_text()).get("finished_at"):
                done.add(res.parent.name.rsplit("__", 1)[0])
        except ValueError:
            pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="served model name (base, swe, term, sci, merged-...)")
    ap.add_argument("--api", default="http://localhost:18000/v1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--jobs-dir", default=str(BASE / "tb2-jobs"))
    ap.add_argument("--reclaim", action="store_true", help="run claimed seeds too (resume after a crash)")
    ap.add_argument("--part", type=int, default=1, help=">1: continue a seed in a new job on unfinished tasks")
    a = ap.parse_args()
    env = dict(os.environ, APPTAINER_CACHEDIR=str(BASE / "apptainer-cache"),
               APPTAINER_TMPDIR=str(BASE / "apptainer-tmp"), HOSTED_VLLM_API_KEY="EMPTY")
    cfg_dir = REPO / "configs/eval/tb2_jobs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        # Several runners (one per serving GPU) share the seed list; each seed is claimed once.
        claim = Path(a.jobs_dir) / f"tb2-{a.name}-s{seed}.claim"
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{a.api} {os.uname().nodename} {os.getpid()}\n".encode())
            os.close(fd)
        except FileExistsError:
            if not a.reclaim:
                print("seed already claimed:", claim, flush=True)
                continue
        cfg = job_config(a.name, a.api, seed, a.n, Path(a.jobs_dir))
        if a.part > 1:
            # Continuation part (e.g. after changing concurrency, which harbor cannot resume): same
            # agent/env settings, only tasks with no finished trial in earlier parts of this seed.
            done = finished_tasks(Path(a.jobs_dir), a.name, seed)
            sub = BASE / "tb2-parts" / f"{a.name}-s{seed}-p{a.part}"
            sub.mkdir(parents=True, exist_ok=True)
            src = REPO / "data/bench/tb2/terminal-bench"
            for t in sorted(src.iterdir()):
                if t.name not in done and not (sub / t.name).exists():
                    (sub / t.name).symlink_to(t)
            cfg["datasets"] = [{"path": str(sub)}]
            cfg["job_name"] = f"tb2-{a.name}-s{seed}-p{a.part}"
            print(f"part {a.part}: {len(list(sub.iterdir()))} unfinished tasks", flush=True)
        path = cfg_dir / f"{cfg['job_name']}.json"
        path.write_text(json.dumps(cfg, indent=1))
        print("running", path, flush=True)
        subprocess.run([str(BASE / "venvs/harbor/bin/harbor"), "run", "-c", str(path), "-y"], env=env, check=False)


if __name__ == "__main__":
    main()
