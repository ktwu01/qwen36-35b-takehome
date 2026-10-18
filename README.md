# Multi-expert training and model merging on Qwen3.5-9B

Candidate repository for the take-home "多专家训练与模型合版实验" (template:
[apexin-ai/qwen36-35b-takehome](https://github.com/apexin-ai/qwen36-35b-takehome)). Three LoRA experts
are trained on public data, each aimed at a subset of SWE-bench Pro v2-hard, Terminal-Bench 2.0, HLE and
ASI-Bench, merged into one checkpoint with task arithmetic / TIES / DARE-TIES, and compared against the
untouched base on all four benchmarks under fixed, identical protocols.

- Candidate repo: https://github.com/ktwu01/qwen36-35b-takehome (this repository)
- Hugging Face checkpoints: see [models/README.md](models/README.md) (links and revisions)
- One-page report: [report.pdf](report.pdf)

> Status markers in this file: every number is produced by `scripts/summarize.py` from the raw runs in
> `results/` and `logs/`. Anything not yet measured is written as "pending", never estimated.

## 1. Research question

Can three narrow experts, each trained in the exact agent protocol of its target benchmark, be merged
into a single model that keeps (most of) each expert's gain without losing the base model's general
ability, and does a merge beat simply training one adapter on the union of the data?

Design hypothesis: SWE-bench Pro and Terminal-Bench need two *different* agent protocols
(mini-swe-agent text actions vs Terminus-2 JSON keystrokes), which would interfere inside one adapter;
HLE and ASI-Bench share the scientific-knowledge domain. Weight-space merging should keep the protocols
apart better than data mixing. ASI-Bench is the interesting case: it is run with the same mini-swe-agent
harness as SWE-Pro but needs science knowledge, so only a model that has both (the merge) can combine
them.

## 2. Base model choice (7B-class fallback, disclosed)

The exam's preferred base is `Qwen/Qwen3.6-35B-A3B` (revision `995ad96eacd98c81ed38be0c5b274b04031597b0`,
downloaded and verified, 71.9 GB bf16). It was not used, because of compute available inside the
deadline:

| Resource | What was available |
|---|---|
| seismic18 | 1x NVIDIA RTX PRO 4000 Blackwell 24 GB, 64 CPU, 502 GB RAM |
| seismic15 / seismic17 | 1x RTX A4500 20 GB each |
| an HPC supercomputer | A100 40 GB nodes via `gpu-a100-dev` (2 h jobs) / `gpu-a100`; GPU queues had 1-2 day waits on 2026-10-18 |

35B-A3B needs ~72 GB just for bf16 weights (all experts are resident even though only ~3B are active),
so neither training nor 256K-context serving fits any single GPU here, and the four full evaluations of
>= 5 models need several GPUs for a day. The exam allows a 7B-class same-family fallback; we use
**`Qwen/Qwen3.5-9B`** (revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), which shares Qwen3.6's
architecture family (`qwen3_5`: Gated DeltaNet + gated attention hybrid, 3:1 linear/full attention,
natively multimodal, 262K context). That keeps the 256K Terminal-Bench setting and HLE's image questions
in scope, which Qwen3-8B (32K, text-only) would not. Tokenizer and chat template: the pinned revision's.
Thinking mode: on (model default) for every model and benchmark.

## 3. Experts and training

| Expert | Target benchmark(s) | Data (license) | Train / dev | Protocol |
|---|---|---|---|---|
| `swe` | SWE-bench Pro v2-hard | SWE-smith mini-swe-agent trajectories (MIT) | 2,000 / 200 | mini-swe-agent text actions |
| `term` | Terminal-Bench 2.0 | OpenThoughts-Agent v1 SFT, Terminus-2 traces (Apache-2.0) | 2,000 / 200 | Terminus-2 JSON |
| `sci` | HLE, ASI-Bench | OpenScienceReasoning-2 (CC-BY-4.0) | 4,000 / 200 | single-turn reasoning |
| control `union` | all | the three sets concatenated | 8,000 / 600 | mixed |

Data construction, cleaning, dedup, difficulty tiers and decontamination: [data/README.md](data/README.md).

All experts share one configuration (`configs/experiment.yaml`), seed `20261018`:

| Setting | Value |
|---|---|
| Method | QLoRA: nf4 4-bit base (double quant), bf16 compute, LoRA adapters in bf16 |
| Why QLoRA | 9B bf16 weights (~19 GB) plus 16K-token activations do not fit 20-40 GB GPUs; peak with QLoRA 18-19 GB |
| Target modules | language model only: full-attention `q/k/v/o_proj` (8 layers), Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj` (24 layers), MLP `gate/up/down_proj` (32 layers); vision tower frozen |
| Rank / alpha / dropout | 32 / 64 / 0.0 (80.2M trainable params, 0.89%) |
| LR / schedule | 1e-4, cosine, 3% warmup |
| Batch | 1 x grad-accum 16 = 16 sequences (union: 3 GPUs x 1 x 5 = 15) |
| Context | 16,384 tokens; longer examples dropped |
| Loss | assistant tokens only (chat-template spans), token-level mean over the accumulated batch |
| Epochs | 1 |
| Checkpoints | every 25 steps, last 3 kept; resumable across 2 h jobs; final adapter = end of epoch |

Hardware, time, throughput, memory and adapter size per expert: pending (`logs/<run_id>/resources.csv`).

## 4. Merging

Implemented in `src/takehome/merge.py` (no external merge library; mergekit pins conflict with the vLLM
stack). Each expert's task vector is the dense LoRA delta `dW_i = (alpha/r) B_i A_i` per adapted matrix;
untouched tensors (embeddings, norms, vision tower, MTP head) are copied bit-exact from the base.

| Method | Rule | Conflict handling |
|---|---|---|
| Task arithmetic | `W + lam * sum_i dW_i` | none (deltas add) |
| TIES | trim each `dW_i` to its top-`density` magnitudes, elect the sign of the summed trimmed deltas per entry, average only agreeing entries | sign election + disjoint mean |
| DARE-TIES | random drop with prob `1 - density`, rescale `1/density`, then TIES election | stochastic sparsification + sign election |

Ablation grid (`configs/merge_ablation.tsv`): task arithmetic lam in {0.5, 0.7, 1.0}; TIES
(density, lam) in {(0.2, 1.0), (0.5, 1.0), (0.5, 0.7)}; DARE-TIES (0.5, 1.0). Selection uses only the
independent dev sets (per-domain token loss, `src/takehome/dev_loss.py`); per-module sign-conflict rates
and pairwise task-vector cosines are saved as `merge_stats.json`. Results: pending.

## 5. Benchmarks (locked before training)

Lock files with source, revision, split and the full task-ID list: `configs/eval/*.lock.json`.

| Benchmark | Version / split | Tasks | Harness | Primary metric |
|---|---|---|---|---|
| SWE-bench Pro v2-hard | `ScaleAI/SWE-bench_Pro` config `hard` @ `2d52cb3`, = `v2/hard51_ids.txt` of `scaleapi/SWE-bench_Pro-os@66f9276` (v2.0.0) | 51 | official v2 Harbor tasks, host mini-swe-agent with the official `mini_textbased.yaml`, 50-min budget, `patch_replay` re-grade on pristine images | resolved rate |
| Terminal-Bench 2.0 | Harbor registry `terminal-bench@2.0`, task digests in the lock | 89 | Harbor 0.24.0 + Terminus-2 | success rate, mean and std over 5 seeds |
| HLE | `cais/hle` test @ `5a81a4c`, all questions incl. images, no tools | 2,500 | official prompts of `centerforaisafety/hle@22ed307`, judge `o3-mini-2025-01-31` | accuracy |
| ASI-Bench | `asibench==0.2.0`, tasks `apexin-ai/ASI-Bench@684ac2c`, instances `ASI-Bench-seed31415` @ `0fa1421`, levels B1-B4 | 59 x 4 | file-exchange mini-swe-agent, `linux_ns` sandbox, 3 h per instance | official local `mean_percent` (+ all-instances mean, per level) |

Shared serving (`scripts/serve.sh`, identical for every model): vLLM 0.31.0, online FP8 weights,
FP8 (e4m3) KV cache, 262,144-token context, prefix caching, thinking on, `qwen3` reasoning parser.
Sampling for every benchmark: temperature 1.0, top_p 0.95, top_k 20 (Qwen thinking defaults, also the
Terminal-Bench model-card setting). Max tokens: TB2 80K (81,920); HLE 32,768; agents default per turn.

### Deviations from the official setups (all apply identically to every evaluated model)

- **Containers: apptainer, not Docker.** The hosts have no Docker and no `/etc/subuid`, so Harbor's
  `singularity` backend is used with a patched bootstrap and launch (`patches/`, installed by
  `scripts/setup_harbor_rt.sh`, frozen at git tag `env-v4-frozen`): each trial runs in its own writable
  reflink copy of the image (`--writable --fakeroot`), a read-only host runtime provides the exec
  server's Python and a static tmux, `/etc/hosts` and `/etc/resolv.conf` are bound in as Docker does,
  the image's `PATH` order is kept, `HOME=/root`, and musl (Alpine) images skip apptainer's glibc
  `faked`. Validation: the reference solutions (oracle) and empty patches (nop) are run through the same
  environment; tasks the oracle cannot pass are counted as environment failures (pending).
- **CPU limit not enforced.** The apptainer backend has no cgroups; the 48 GB memory cap is enforced by
  Harbor's watchdog, 32 CPUs is requested but not enforced.
- **SWE-Pro agent phase not network-isolated.** The locked protocol's `no-network` agent phase cannot be
  enforced by this backend; `configs/eval/swe_pro_task_modifications.json` records the one-line change
  per task. Trajectories are audited for network use (pending).
- **Serving hardware differs across hosts** (RTX PRO 4000 Blackwell, RTX A4500, A100) with identical
  vLLM flags and quantization; which host served each run is recorded in the claim files and logs.

## 6. Results

Pending. The table below is filled by `scripts/summarize.py` (`results/summary.csv`).

| Benchmark | Base | swe | term | sci | Merged | Merged - Base (pp, 95% CI) |
|---|---|---|---|---|---|---|
| SWE-bench Pro v2-hard | pending | pending | n/e | n/e | pending | pending |
| Terminal-Bench 2.0 | pending | n/e | pending | n/e | pending | pending |
| HLE | pending | n/e | n/e | pending | pending | pending |
| ASI-Bench | pending | n/e | n/e | pending | pending | pending |
| Macro average | pending | | | | pending | pending |

n/e: not evaluated (experts are evaluated on their target benchmarks).

## 7. Reproduction

```bash
# 0. environments (local disk), harbor runtime + patches
uv venv -p 3.12 /var/tmp/$USER/venvs/train && uv pip install -p /var/tmp/$USER/venvs/train/bin/python -r requirements.lock
bash scripts/setup_harbor_rt.sh && bash scripts/setup_serving.sh
# 1. lock benchmarks, build data
python scripts/lock_benchmarks.py
PYTHONPATH=src python -m takehome.build_contam
for e in swe term; do PYTHONPATH=src python -m takehome.build_data --expert $e --n-train 2000; done
PYTHONPATH=src python -m takehome.build_data --expert sci --n-train 4000
PYTHONPATH=src python -m takehome.build_data --expert union
# 2. train (HPC) and merge-ablate on dev sets
sbatch scripts/hpc_train_experts.slurm; sbatch scripts/hpc_merge_ablation.slurm; sbatch scripts/hpc_train_union.slurm
# 3. evaluate a served model (repeat per model)
bash scripts/serve.sh <path> <name> 18000
python scripts/run_tb2.py --name <name> --seeds 1 2 3 4 5
bash scripts/run_swepro.sh <name> http://localhost:18000/v1
PYTHONPATH=src python -m takehome.eval_hle generate --name <name> --out results/<group>/hle/<name>/run1
PYTHONPATH=src python -m takehome.eval_hle judge --out results/<group>/hle/<name>/run1
bash scripts/run_asi.sh <name> http://localhost:18000/v1
# 4. tables, CIs, plots
python scripts/summarize.py --base /var/tmp/$USER
```

## 8. Known limitations

- 7B-class fallback instead of the 35B-A3B base (section 2).
- Container environment differs from official Docker/Modal runs (section 5); official scores are not
  claimed and not comparable.
- One training seed per expert; repeat-run variance is measured only on Terminal-Bench (5 seeds).
