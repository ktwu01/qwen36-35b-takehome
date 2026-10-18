"""Post-hoc HLE contamination check of the training sets (HLE access arrived after data was built).

    python scripts/hle_contam_check.py      # after `python -m takehome.build_contam` included HLE

Counts training/dev examples whose prompt shares any 13-word n-gram with an HLE question, per expert.
Writes data/manifests/hle_contamination.json. (Same rule as build_data's decontamination.)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takehome.build_data import NGRAM, words  # noqa: E402

grams = set()
n_hle = 0
for line in open("data/bench/contam_texts.jsonl"):
    r = json.loads(line)
    if r["bench"] != "hle":
        continue
    n_hle += 1
    w = words(r["text"])
    grams.update(" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1))
assert n_hle, "HLE texts missing: rerun takehome.build_contam with HLE access"
out = {"hle_questions": n_hle, "experts": {}}
for e in ("swe", "term", "sci"):
    hits = {}
    for split in ("train", "dev"):
        ids = []
        for line in open(f"data/processed/{e}/{split}.jsonl"):
            r = json.loads(line)
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            w = words(user)
            if any(" ".join(w[i:i + NGRAM]) in grams for i in range(len(w) - NGRAM + 1)):
                ids.append(r["id"])
        hits[split] = {"n_overlapping": len(ids), "ids": ids}
    out["experts"][e] = hits
    print(e, {s: v["n_overlapping"] for s, v in hits.items()})
Path("data/manifests/hle_contamination.json").write_text(json.dumps(out, indent=1))
