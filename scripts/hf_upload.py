"""Upload an expert adapter / union adapter / merged model to a public Hugging Face model repo.

    python scripts/hf_upload.py --kind adapter --name expert-swe --path outputs/swe/adapter
    python scripts/hf_upload.py --kind merged --name merged --path /var/tmp/$USER/models/merged \
        --recipe results/dev/<chosen>.recipe.json

Writes a model card (README.md) next to the weights, creates ktwu01/qwen35-9b-takehome-<name> as public,
uploads, and records {repo, url, revision} in models/hf_index.json for models/README.md.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

OWNER = "ktwu01"
BASE, BASE_REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DATA = {
    "expert-swe": ["Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k (MIT)"],
    "expert-term": ["open-thoughts/OpenThoughts-Agent-v1-SFT (Apache-2.0)"],
    "expert-sci": ["nvidia/OpenScienceReasoning-2 (CC-BY-4.0, NVIDIA)"],
}
DATA["union-control"] = DATA["expert-swe"] + DATA["expert-term"] + DATA["expert-sci"]
DATA["merged"] = DATA["union-control"]
TARGET = {"expert-swe": "SWE-bench Pro v2-hard", "expert-term": "Terminal-Bench 2.0",
          "expert-sci": "HLE and ASI-Bench", "union-control": "all four (control: one adapter on the union of the data)",
          "merged": "all four (SWE-bench Pro v2-hard, Terminal-Bench 2.0, HLE, ASI-Bench)"}


def card(name, kind, recipe, results_md):
    repo = f"{OWNER}/qwen35-9b-takehome-{name}"
    if kind == "adapter":
        load = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{BASE}", revision="{BASE_REV}", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{repo}")   # pin revision=... from the candidate repo README
tok = AutoTokenizer.from_pretrained("{BASE}", revision="{BASE_REV}")
```"""
    else:
        load = f"""```bash
vllm serve {repo} --quantization fp8 --kv-cache-dtype fp8 --max-model-len 262144 \\
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```
or `AutoModelForCausalLM.from_pretrained("{repo}")` (same architecture and files as the base)."""
    rec = f"\n## Merge recipe\n\n```json\n{json.dumps(recipe, indent=1)}\n```\n" if recipe else ""
    data = "\n".join(f"- {d}" for d in DATA[name])
    return f"""---
license: apache-2.0
base_model: {BASE}
library_name: {"peft" if kind == "adapter" else "transformers"}
tags: [qwen3.5, lora, model-merging, take-home]
---

# qwen35-9b-takehome-{name}

{"LoRA adapter" if kind == "adapter" else "Merged full checkpoint"} on `{BASE}` (revision `{BASE_REV}`),
part of a multi-expert training and model-merging experiment. Target: {TARGET[name]}.
Code, configs, logs and the full protocol: https://github.com/ktwu01/qwen36-35b-takehome

## Training

QLoRA (nf4 base, bf16 compute), r=32, alpha=64, dropout 0, targets: full-attention q/k/v/o_proj,
Gated DeltaNet in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down_proj (language model only), lr 1e-4
cosine, 3% warmup, global batch 16, 16,384-token context, 1 epoch, assistant-only loss,
seed 20261018.

## Data

{data}

Decontaminated with 13-gram overlap against all four benchmarks; see `data/README.md` in the code repo.
{rec}
## Loading

{load}

## Evaluation

{results_md or "See the candidate repository README for results, confidence intervals and protocols."}

## License and limitations

Apache-2.0 (as the base model). Training data licenses above; OpenScienceReasoning-2 is CC-BY-4.0 and
requires attribution to NVIDIA. Research artifact: evaluated under the experiment-specific protocol in the
code repository (apptainer environment, FP8 serving); scores are not official leaderboard results.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["adapter", "merged"], required=True)
    ap.add_argument("--name", required=True, choices=list(DATA))
    ap.add_argument("--path", required=True)
    ap.add_argument("--recipe")
    ap.add_argument("--results-md")
    a = ap.parse_args()
    path = Path(a.path)
    recipe = json.loads(Path(a.recipe).read_text()) if a.recipe else None
    results_md = Path(a.results_md).read_text() if a.results_md else None
    (path / "README.md").write_text(card(a.name, a.kind, recipe, results_md))
    api = HfApi()
    repo = f"{OWNER}/qwen35-9b-takehome-{a.name}"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(folder_path=str(path), repo_id=repo, repo_type="model",
                             commit_message=f"Upload {a.name}", allow_patterns=None,
                             ignore_patterns=["ckpt/*", "*.pt", "optimizer*", "rng_state*", "._*", ".DS_Store"])
    idx_path = Path("models/hf_index.json")
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[a.name] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "revision": info.oid, "kind": a.kind}
    idx_path.parent.mkdir(exist_ok=True)
    idx_path.write_text(json.dumps(idx, indent=1))
    print(json.dumps(idx[a.name]))


if __name__ == "__main__":
    main()
