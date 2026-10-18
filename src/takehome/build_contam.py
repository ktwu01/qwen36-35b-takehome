"""Collect benchmark texts used only for 13-gram decontamination of training data.

    python -m takehome.build_contam

Writes data/bench/contam_texts.jsonl ({"bench", "id", "text"}); git-ignored, never trained on.
Sources: SWE-bench Pro problem statements (all 642 v2 tasks, not just hard), HLE questions,
ASI-Bench prompts b1-b4, Terminal-Bench 2.0 instructions (data/bench/tb2, from lock_benchmarks).
"""

import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def lock(name):
    return json.load(open(f"configs/eval/{name}.lock.json"))


def main():
    out = Path("data/bench/contam_texts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {}
    with open(out, "w") as f:
        def emit(bench, i, text):
            f.write(json.dumps({"bench": bench, "id": i, "text": text}) + "\n")
            n[bench] = n.get(bench, 0) + 1

        sp = lock("swe_pro_v2_hard")
        for r in load_dataset(sp["repo"], "default", split="test", revision=sp["revision"]):
            emit("swe_pro", r["instance_id"], r["problem_statement"] + "\n" + (r["requirements"] or ""))

        hl = Path("configs/eval/hle.lock.json")
        if hl.exists():
            h = lock("hle")
            for r in load_dataset(h["repo"], split=h["split"], revision=h["revision"]).remove_columns(["image"]):
                emit("hle", r["id"], r["question"])
        else:
            print("WARNING: HLE not locked yet; rerun after access is granted")

        a = lock("asi_bench")
        root = Path(snapshot_download(a["repo"], repo_type="dataset", revision=a["revision"], allow_patterns=["tasks/*/prompt_b*.md"]))
        for p in sorted(root.glob("tasks/*/prompt_b*.md")):
            emit("asi", f"{p.parent.name}/{p.stem}", p.read_text())

        tb = Path("data/bench/tb2")
        for p in sorted(tb.rglob("instruction.md")) if tb.exists() else []:
            emit("tb2", p.parent.name, p.read_text())
        if not tb.exists():
            print("WARNING: TB2 not downloaded yet; rerun after lock_benchmarks --only tb2")
    print(out, n)


if __name__ == "__main__":
    main()
