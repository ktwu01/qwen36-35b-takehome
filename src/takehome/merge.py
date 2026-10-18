"""Merge expert LoRA adapters into one full Qwen3.5-9B checkpoint.

    python -m takehome.merge --method ties --experts outputs/swe outputs/term outputs/sci \
        --weights 1 1 1 --density 0.5 --lam 1.0 --out outputs/merged-ties

Every expert is a LoRA adapter on the same base. For each adapted linear W the task vector
is the dense delta  dW_i = (alpha / r) * B_i @ A_i  (fp32). Methods:
  single          W + dW_1                                  (export one expert as full weights)
  task_arithmetic W + lam * sum_i w_i dW_i                  (Ilharco et al., 2023)
  ties            trim each dW_i to its top-`density` magnitudes, elect a per-entry sign by
                  the sign of sum_i w_i trim(dW_i), average only entries agreeing with it,
                  scale by lam                              (Yadav et al., 2023)
  dare_ties       drop each entry of dW_i with prob 1-density, rescale by 1/density, then the
                  TIES sign election and disjoint mean     (Yu et al., 2024)
Untouched tensors (embeddings, norms, vision tower, MTP head) are copied bit-exact from the
base. Writes merge_stats.json with per-module sign-conflict rate and pairwise cosine between
task vectors, plus merge_recipe.json.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_KEY = re.compile(r"base_model\.model\.(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def load_adapter(path):
    path = Path(path)
    path = path / "adapter" if (path / "adapter").is_dir() else path
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    parts = defaultdict(dict)
    with safe_open(path / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            m = ADAPTER_KEY.match(k)
            if not m:
                raise ValueError(f"unexpected adapter key {k}")
            layer, mod, ab = m.groups()
            parts[f"model.language_model.layers.{layer}.{mod}.weight"][ab] = f.get_tensor(k).float()
    return {k: (v["B"] @ v["A"]) * scale for k, v in parts.items()}, cfg


def ties_merge(deltas, weights, density, lam, dare, gen):
    trimmed = []
    for d in deltas:
        if dare:
            keep = torch.rand(d.shape, generator=gen) < density
            t = d * keep / density
        else:
            k = max(1, int(d.numel() * density))
            thr = d.abs().flatten().kthvalue(d.numel() - k + 1).values
            t = d * (d.abs() >= thr)
        trimmed.append(t)
    stack = torch.stack([w * t for w, t in zip(weights, trimmed)])
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign) & (stack != 0)
    num = (stack * agree).sum(0)
    den = agree.sum(0).clamp(min=1)
    return lam * num / den


def conflict_stats(deltas):
    signs = torch.stack([torch.sign(d) for d in deltas])
    nz = (signs != 0).all(0)
    conflict = ((signs != signs[0]).any(0) & nz).float().sum() / nz.float().sum().clamp(min=1)
    flat = [d.flatten() for d in deltas]
    cos = {f"{i}-{j}": torch.nn.functional.cosine_similarity(flat[i], flat[j], dim=0).item()
           for i in range(len(flat)) for j in range(i + 1, len(flat))}
    return {"sign_conflict_rate": conflict.item(), "cosine": cos,
            "norms": [d.norm().item() for d in deltas]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--method", required=True, choices=["single", "task_arithmetic", "ties", "dare_ties"])
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--layers", default="", help="optional 'lo-hi' layer range to merge; others keep base")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = a.weights or [1.0] * len(a.experts)
    assert len(weights) == len(a.experts)
    if a.method == "single":
        assert len(a.experts) == 1
    lo, hi = (map(int, a.layers.split("-")) if a.layers else (0, 10**9))
    gen = torch.Generator().manual_seed(a.seed)

    base_dir = Path(snapshot_download(a.base, revision=a.revision))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adapters = [load_adapter(e)[0] for e in a.experts]
    keys = sorted(set().union(*adapters))
    stats = {}

    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    by_shard = defaultdict(list)
    for k, shard in index["weight_map"].items():
        by_shard[shard].append(k)
    touched = 0
    for shard, names in sorted(by_shard.items()):
        tensors = {}
        with safe_open(base_dir / shard, "pt") as f:
            for name in names:
                w = f.get_tensor(name)
                if name in keys:
                    layer = int(name.split(".layers.")[1].split(".")[0])
                    deltas = [ad.get(name, torch.zeros(w.shape)) for ad in adapters]
                    if lo <= layer <= hi:
                        if a.method == "single":
                            delta = deltas[0]
                        elif a.method == "task_arithmetic":
                            delta = a.lam * sum(wt * d for wt, d in zip(weights, deltas))
                        else:
                            delta = ties_merge(deltas, weights, a.density, a.lam, a.method == "dare_ties", gen)
                        w = (w.float() + delta).to(w.dtype)
                        touched += 1
                    if len(deltas) > 1:
                        stats[name] = conflict_stats(deltas)
                tensors[name] = w.contiguous()
        save_file(tensors, out / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(names)} tensors")
    for p in base_dir.iterdir():
        if not p.name.endswith(".safetensors") and p.is_file():
            shutil.copy(p, out / p.name)
    recipe = {"method": a.method, "experts": a.experts, "weights": weights, "lam": a.lam,
              "density": a.density, "layers": a.layers or "all", "seed": a.seed,
              "base": a.base, "base_revision": a.revision, "modules_merged": touched}
    (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=1))
    if stats:
        (out / "merge_stats.json").write_text(json.dumps(stats))
    print(json.dumps(recipe))


if __name__ == "__main__":
    main()
