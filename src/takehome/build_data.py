"""Build each expert's train/dev set.

    python -m takehome.build_data --expert swe --n-train 2000
    python -m takehome.build_data --expert union      # concatenation of the three built experts

Pipeline per expert: normalize to {"messages"} -> drop over-length -> exact dedup on the
task-specific prompt span (task_text) -> MinHash near-dup (Jaccard >= 0.8 on 5-shingles) -> 13-gram decontamination
against all four benchmarks (data/bench/contam_texts.jsonl, built by build_contam.py) ->
SWE-Pro repo exclusion (swe only) -> difficulty tier -> stratified sample -> dev split by
task group. Writes data/processed/<expert>/{train,dev}.jsonl and data/manifests/<expert>.json
with the count removed by every rule and the sha256 of each output.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

SOURCES = {
    "swe": ("Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k", "MIT"),
    "term": ("open-thoughts/OpenThoughts-Agent-v1-SFT", "Apache-2.0"),
    "sci": ("nvidia/OpenScienceReasoning-2", "CC-BY-4.0"),
}
MAX_CHARS = 60_000      # ~16K tokens; exact token filter is applied again in sft.py
NGRAM = 13
DEV_FRACTION = 0.04


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def words(s):
    return re.findall(r"\w+", s.lower())


def load_expert(name):
    repo, _ = SOURCES[name]
    ds = load_dataset(repo, split="train")
    out = []
    if name == "swe":
        for r in ds:
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            out.append({"id": r["instance_id"], "group": r["instance_id"].split(".")[0], "messages": msgs, "prompt": user})
    elif name == "term":
        for r in ds:
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["conversations"]]
            out.append({"id": r["trial_name"], "group": r["task"], "messages": msgs, "prompt": msgs[0]["content"]})
    elif name == "sci":
        for i, r in enumerate(ds):
            msgs = [{"role": "user", "content": r["input"]}, {"role": "assistant", "content": r["output"]}]
            out.append({"id": f"osr2-{i}", "group": f"osr2-{i}", "messages": msgs, "prompt": r["input"]})
    return out


def task_text(name, prompt):
    """Task-specific span of the prompt, used for dedup (agent prompts share long templates)."""
    if name == "term":
        a, b = prompt.find("Task Description"), prompt.find("Current terminal state")
        return prompt[a:b] if a >= 0 and b > a else prompt
    if name == "swe":
        a, b = prompt.find("<pr_description>"), prompt.find("</pr_description>")
        return prompt[a:b] if a >= 0 and b > a else prompt
    return prompt


def tier(ex):
    n_asst = sum(m["role"] == "assistant" for m in ex["messages"])
    size = sum(len(m["content"]) for m in ex["messages"])
    if n_asst > 1:
        return "hard" if n_asst > 25 else "medium" if n_asst > 10 else "easy"
    return "hard" if size > 24_000 else "medium" if size > 8_000 else "easy"


def contam_index(path):
    grams = set()
    for line in open(path):
        w = words(json.loads(line)["text"])
        grams.update(" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1))
    return grams


def is_contaminated(text, grams):
    w = words(text)
    return any(" ".join(w[i:i + NGRAM]) in grams for i in range(len(w) - NGRAM + 1))


def minhash(text):
    m = MinHash(num_perm=64)
    w = words(text)
    for sh in {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}:
        m.update(sh.encode())
    return m


def swe_pro_repo_keys(lock="configs/eval/swe_pro_v2_hard.lock.json"):
    """'ansible/ansible' -> {'ansible__ansible', 'ansible'} so SWE-smith ids match either form."""
    keys = set()
    for repo in json.load(open(lock))["excluded_repos_for_training"]:
        owner, name = repo.split("/")
        keys |= {f"{owner}__{name}".lower(), name.lower()}
    return keys


def build(name, n_train, seed, contam_path):
    rng = random.Random(seed)
    rows = load_expert(name)
    stats = Counter(raw=len(rows))

    rows = [r for r in rows if sum(len(m["content"]) for m in r["messages"]) <= MAX_CHARS]
    stats["after_length"] = len(rows)

    seen, uniq = set(), []
    for r in rows:
        h = hashlib.sha1(norm(task_text(name, r["prompt"])).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    rows = uniq
    stats["after_exact_dedup"] = len(rows)

    if name == "swe":
        bad = swe_pro_repo_keys()
        rows = [r for r in rows if r["group"].lower() not in bad and r["group"].split("__")[-1].lower() not in bad]
        stats["after_swe_pro_repo_exclusion"] = len(rows)

    grams = contam_index(contam_path)
    rows = [r for r in rows if not is_contaminated(r["prompt"], grams)]
    stats["after_13gram_decontam"] = len(rows)

    # Near-dup only on a shuffled working pool (MinHash over 1M+ sci rows is not needed).
    rng.shuffle(rows)
    pool = rows[: max(4 * n_train, 20_000)]
    lsh, kept = MinHashLSH(threshold=0.8, num_perm=64), []
    for r in pool:
        m = minhash(task_text(name, r["prompt"])[:4000])
        if not lsh.query(m):
            lsh.insert(r["id"], m)
            kept.append(r)
    stats["pool_after_near_dedup"] = len(kept)
    stats["near_dup_removed_in_pool"] = len(pool) - len(kept)

    for r in kept:
        r["tier"] = tier(r)
    groups = sorted({r["group"] for r in kept})
    rng.shuffle(groups)
    dev_groups = set(groups[: max(1, int(len(groups) * DEV_FRACTION))])
    dev = [r for r in kept if r["group"] in dev_groups][:200]
    train_pool = [r for r in kept if r["group"] not in dev_groups]

    # Stratified by tier: equal thirds when available, rest filled from the remaining pool.
    by_tier = {t: [r for r in train_pool if r["tier"] == t] for t in ("easy", "medium", "hard")}
    train = []
    for t, lst in by_tier.items():
        train += lst[: n_train // 3]
    taken = {r["id"] for r in train}
    rest = [r for r in train_pool if r["id"] not in taken] if len(train) < n_train else []
    train += rest[: n_train - len(train)]
    rng.shuffle(train)
    stats.update(train=len(train), dev=len(dev))
    return train, dev, stats, Counter(r["tier"] for r in train)


def write(name, train, dev, stats, tiers, extra=None):
    d = Path("data/processed") / name
    d.mkdir(parents=True, exist_ok=True)
    shas = {}
    for split, rows in (("train", train), ("dev", dev)):
        p = d / f"{split}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "tier": r.get("tier"), "messages": r["messages"]}, ensure_ascii=False) + "\n")
        shas[split] = hashlib.sha256(p.read_bytes()).hexdigest()
        (d / f"{split}.ids").write_text("\n".join(r["id"] for r in rows) + "\n")
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    man = {"expert": name, "source": SOURCES.get(name), "filters": dict(stats), "train_tiers": dict(tiers),
           "sha256": shas, **(extra or {})}
    json.dump(man, open(f"data/manifests/{name}.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, choices=[*SOURCES, "union"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--contam", default="data/bench/contam_texts.jsonl")
    a = ap.parse_args()
    if a.expert == "union":
        train, dev = [], []
        for e in SOURCES:
            d = Path("data/processed") / e
            train += [json.loads(l) for l in open(d / "train.jsonl")]
            dev += [json.loads(l) for l in open(d / "dev.jsonl")]
        random.Random(a.seed).shuffle(train)
        write("union", train, dev, Counter(train=len(train), dev=len(dev)), Counter(r["tier"] for r in train),
              {"source": "concatenation of swe, term, sci train/dev"})
        return
    write(a.expert, *build(a.expert, a.n_train, a.seed, a.contam))


if __name__ == "__main__":
    main()
