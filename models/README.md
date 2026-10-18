# Models

All checkpoints are public on Hugging Face (Apache-2.0). Base: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the original release; not re-uploaded). Tokenizer and chat template: the base revision's. Uploaded and verified anonymously downloadable by `scripts/hf_upload.py`.

| Model | Kind | Hugging Face | Revision |
|---|---|---|---|
| SWE expert (target: SWE-bench Pro v2-hard) | adapter | [ktwu01/qwen35-9b-takehome-expert-swe](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-swe) | `387687daaec0db58727cb70ee59c7acb49c83d4c` |
| Terminal expert (target: Terminal-Bench 2.0) | adapter | [ktwu01/qwen35-9b-takehome-expert-term](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-term) | `16ae50eb52fc7458476fce134cecee107e909d13` |
| Science expert (targets: HLE, ASI-Bench) | adapter | [ktwu01/qwen35-9b-takehome-expert-sci](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-sci) | `0e9ea7a01100f1a61f5a0eff7bf81eb52e1ede86` |

## Loading

LoRA adapters (experts, control):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "ktwu01/qwen35-9b-takehome-expert-term", revision="<revision above>")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
```

Full checkpoints exported from an adapter (what the evaluations served) are reproduced exactly with
`python -m takehome.merge --method single --experts <adapter dir> --out <dir>`.

Merged model (full weights, same architecture and files as the base):

```bash
vllm serve ktwu01/qwen35-9b-takehome-merged --revision <revision above> --quantization fp8 \
  --kv-cache-dtype fp8 --max-model-len 262144 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

## Licenses

Weights: Apache-2.0 (as the base). Training data: SWE-smith trajectories (MIT),
OpenThoughts-Agent v1 SFT (Apache-2.0), OpenScienceReasoning-2 (CC-BY-4.0, attribution to NVIDIA).
