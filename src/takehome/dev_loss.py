"""Dev-set loss of a full checkpoint (base, exported expert, or merged model) per domain.

    python -m takehome.dev_loss --model outputs/merged/ties-d0.5-l1.0 --name ties-d0.5-l1.0 \
        --out results/dev/ties-d0.5-l1.0.json

Used to compare merge methods/configs and the union control without touching any benchmark.
Same tokenization and assistant-only masking as training (takehome.sft.encode), the same first
100 examples of data/processed/<domain>/dev.jsonl, bf16 weights, chunked cross-entropy.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from takehome.sft import encode

DOMAINS = ("swe", "term", "sci")


@torch.no_grad()
def domain_loss(model, tok, path, n, max_length, chunk=1024):
    total, count, per = 0.0, 0, []
    for line in list(open(path))[:n]:
        e = encode(tok, json.loads(line)["messages"], max_length)
        if e is None:
            continue
        ids = torch.tensor([e["input_ids"]], device=model.device)
        lab = torch.tensor([e["labels"]], device=model.device)[:, 1:]
        h = model.model(input_ids=ids).last_hidden_state[:, :-1]
        keep = lab != -100
        hs, ys = h[keep], lab[keep]
        s = sum(torch.nn.functional.cross_entropy(model.lm_head(hs[i:i + chunk]).float(), ys[i:i + chunk],
                                                  reduction="sum").item() for i in range(0, len(ys), chunk))
        total, count = total + s, count + len(ys)
        per.append(s / max(1, len(ys)))
    return {"token_loss": total / max(1, count), "n_examples": len(per), "n_tokens": count, "per_example": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    res = {"name": a.name, "model": a.model, "domains": {}}
    for d in DOMAINS:
        res["domains"][d] = domain_loss(model, tok, f"data/processed/{d}/dev.jsonl", a.n, a.max_length)
        print(a.name, d, round(res["domains"][d]["token_loss"], 4), flush=True)
    res["mean_token_loss"] = sum(v["token_loss"] for v in res["domains"].values()) / len(DOMAINS)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
