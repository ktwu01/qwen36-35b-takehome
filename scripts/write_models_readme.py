"""Render models/README.md from models/hf_index.json (written by scripts/hf_upload.py).

    python scripts/write_models_readme.py
"""

import json
from pathlib import Path

BASE, BASE_REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ORDER = ["expert-swe", "expert-term", "expert-sci", "merged", "union-control"]
ROLE = {"expert-swe": "SWE expert (target: SWE-bench Pro v2-hard)",
        "expert-term": "Terminal expert (target: Terminal-Bench 2.0)",
        "expert-sci": "Science expert (targets: HLE, ASI-Bench)",
        "merged": "Final merged model (all four benchmarks)",
        "union-control": "Control: one adapter on the union of the three datasets"}

idx = json.loads(Path("models/hf_index.json").read_text())
recipe = Path("configs/merge_final.json")
lines = ["# Models", "",
         f"All checkpoints are public on Hugging Face (Apache-2.0). Base: [`{BASE}`](https://huggingface.co/{BASE}) "
         f"at revision `{BASE_REV}` (the original release; not re-uploaded). Tokenizer and chat template: the "
         "base revision's. Uploaded and verified anonymously downloadable by `scripts/hf_upload.py`.", "",
         "| Model | Kind | Hugging Face | Revision |", "|---|---|---|---|"]
for k in ORDER:
    if k in idx:
        e = idx[k]
        lines.append(f"| {ROLE[k]} | {e['kind']} | [{e['repo']}]({e['url']}) | `{e['revision']}` |")
lines += ["", "## Loading", "", "LoRA adapters (experts, control):", "", "```python",
          "from transformers import AutoModelForCausalLM, AutoTokenizer",
          "from peft import PeftModel",
          f'base = AutoModelForCausalLM.from_pretrained("{BASE}", revision="{BASE_REV}", dtype="bfloat16")',
          'model = PeftModel.from_pretrained(base, "ktwu01/qwen35-9b-takehome-expert-term", revision="<revision above>")',
          f'tok = AutoTokenizer.from_pretrained("{BASE}", revision="{BASE_REV}")', "```", "",
          "Full checkpoints exported from an adapter (what the evaluations served) are reproduced exactly with",
          "`python -m takehome.merge --method single --experts <adapter dir> --out <dir>`.", "",
          "Merged model (full weights, same architecture and files as the base):", "", "```bash",
          "vllm serve ktwu01/qwen35-9b-takehome-merged --revision <revision above> --quantization fp8 \\",
          "  --kv-cache-dtype fp8 --max-model-len 262144 --reasoning-parser qwen3 \\",
          "  --enable-auto-tool-choice --tool-call-parser qwen3_coder", "```", ""]
if recipe.exists():
    lines += ["## Merge recipe (final model)", "", "```json", recipe.read_text().strip(), "```", ""]
lines += ["## Licenses", "", "Weights: Apache-2.0 (as the base). Training data: SWE-smith trajectories (MIT),",
          "OpenThoughts-Agent v1 SFT (Apache-2.0), OpenScienceReasoning-2 (CC-BY-4.0, attribution to NVIDIA).", ""]
Path("models/README.md").write_text("\n".join(lines))
print("\n".join(lines[:12]))
