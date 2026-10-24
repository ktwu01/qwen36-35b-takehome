# Models

All checkpoints are public on Hugging Face (Apache-2.0). Base: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the original release; not re-uploaded). Tokenizer and chat template: the base revision's. Uploaded and verified anonymously downloadable by `scripts/hf_upload.py`.

| Model | Kind | Hugging Face | Revision |
|---|---|---|---|
| SWE expert (target: SWE-bench Pro v2-hard) | adapter | [ktwu01/qwen35-9b-takehome-expert-swe](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-swe) | `6e499da98f37709fbe904d16428ad860775075ec` |
| Terminal expert (target: Terminal-Bench 2.0) | adapter | [ktwu01/qwen35-9b-takehome-expert-term](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-term) | `3cde28a537c70f9f26a7fe3f0fbc89f946a67f30` |
| Science expert (targets: HLE, ASI-Bench) | adapter | [ktwu01/qwen35-9b-takehome-expert-sci](https://huggingface.co/ktwu01/qwen35-9b-takehome-expert-sci) | `cd65629977f9dd36db71a955c6c45d8ed82459dd` |
| Final merged model (all four benchmarks) | merged | [ktwu01/qwen35-9b-takehome-merged](https://huggingface.co/ktwu01/qwen35-9b-takehome-merged) | `e1f112a11ffb987084e87cf6cd1d47a2b93e7520` |
| Control: one adapter on the union of the three datasets | adapter | [ktwu01/qwen35-9b-takehome-union-control](https://huggingface.co/ktwu01/qwen35-9b-takehome-union-control) | `2b9606c29428cced9cf3f6cd6419d46115ef9f7a` |

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

## Merge recipe (final model)

```json
{
 "name": "ties-d0.5-l1.0",
 "merge_args": "--method ties --experts outputs/swe outputs/term outputs/sci --density 0.5 --lam 1.0",
 "dev_mean_token_loss": 0.32781994345410975,
 "dev_token_loss": {
  "swe": 0.22026807381601063,
  "term": 0.1873204659077618,
  "sci": 0.5758712906385569
 },
 "rule": "lowest mean dev token loss over swe/term/sci among 3-expert merges; tie -> lower worst domain",
 "ranking": [
  {
   "name": "ties-d0.5-l1.0",
   "mean": 0.32782,
   "worst": 0.57587
  },
  {
   "name": "ties-d0.2-l1.0",
   "mean": 0.32953,
   "worst": 0.57779
  },
  {
   "name": "ties-d0.5-l0.7",
   "mean": 0.3317,
   "worst": 0.57579
  },
  {
   "name": "ta-l0.5",
   "mean": 0.33197,
   "worst": 0.58267
  },
  {
   "name": "dare-d0.5-l1.0",
   "mean": 0.3363,
   "worst": 0.58989
  },
  {
   "name": "ta-l0.7",
   "mean": 0.33852,
   "worst": 0.59366
  },
  {
   "name": "ta-l1.0",
   "mean": 0.36116,
   "worst": 0.62241
  }
 ]
}
```

## Licenses

Weights: Apache-2.0 (as the base). Training data: SWE-smith trajectories (MIT),
OpenThoughts-Agent v1 SFT (Apache-2.0), OpenScienceReasoning-2 (CC-BY-4.0, attribution to NVIDIA).
