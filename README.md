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
independent dev sets (per-domain token loss on the first 100 dev examples, `src/takehome/dev_loss.py`,
bf16 weights, one A100) under a rule fixed before any result (`scripts/select_merge.py`): lowest mean
dev loss among 3-expert merges. Per-module sign-conflict rates and pairwise task-vector cosines are in
`results/dev/*.merge_stats.json`; plot `curves/merge_ablation.png`, data `curves/source/merge_ablation_dev_loss.csv`.

| Model / merge | swe dev | term dev | sci dev | mean |
|---|---|---|---|---|
| Base | 0.2907 | 0.3082 | 0.6242 | 0.4077 |
| expert swe | **0.2083** | 0.2574 | 0.6304 | 0.3654 |
| expert term | 0.2377 | **0.1706** | 0.6260 | 0.3448 |
| expert sci | 0.2880 | 0.2801 | **0.5619** | 0.3767 |
| task arithmetic lam 1.0 | 0.2535 | 0.2076 | 0.6224 | 0.3612 |
| task arithmetic lam 0.7 | 0.2311 | 0.1908 | 0.5937 | 0.3385 |
| task arithmetic lam 0.5 | 0.2225 | 0.1908 | 0.5827 | 0.3320 |
| DARE-TIES d 0.5 lam 1.0 | 0.2292 | 0.1898 | 0.5899 | 0.3363 |
| TIES d 0.5 lam 0.7 | 0.2207 | 0.1987 | 0.5758 | 0.3317 |
| TIES d 0.2 lam 1.0 | 0.2214 | 0.1894 | 0.5778 | 0.3295 |
| **TIES d 0.5 lam 1.0 (selected, `configs/merge_final.json`)** | 0.2203 | 0.1873 | 0.5759 | **0.3278** |

What the merge does on dev: it keeps 85% (swe), 88% (term) and 78% (sci) of each expert's own-domain
loss reduction relative to base, inside one model. The task vectors are nearly orthogonal (mean cosine
swe-term 0.097, swe-sci 0.030, term-sci 0.026) and the per-entry sign-conflict rate among non-zero
entries is 0.73, close to the 0.75 expected for three independent sign patterns; so plain summation
(task arithmetic lam 1.0) over-shoots and erases most of the science gain, scaling down helps, and TIES'
per-entry sign election plus disjoint mean handles the conflicts best. DARE's random drop is worse than
TIES' magnitude trim at the same density.

### Controls

- **Number of experts merged** (dev loss): merging only the two agent experts (TIES swe+term d 0.5)
  gives slightly better agent losses (swe 0.2151, term 0.1803) than the 3-expert merge (0.2203, 0.1873)
  but science gets worse than base (0.6312 vs 0.6242). Adding the science expert costs ~0.005-0.007 on the
  agent domains and buys a 0.055 science improvement.
- **Training strength** (dev loss): term expert at 60% of training (checkpoint 75) vs final: term 0.1761
  vs 0.1706. (The equivalent swe checkpoint-50 had been rotated out by `save_total_limit`; not run.)
- **Union of the data in one adapter** (planned "merge vs single model"): attempted as a 3-GPU DDP run,
  failed at the first step (`DistributedDataParallel` hides `get_base_model`, used by the chunked loss;
  `logs/train-union/`), and was dropped for time. Not reported as a result.

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
- **mini-swe-agent action fence.** SWE-Pro and ASI-Bench use the official text-based mini-swe-agent
  configs with one change, `mswea_bash_command` -> `bash` in the templates and `model.action_regex`
  (`scripts/make_swepro_config.py`): mini-swe-agent v2 renamed the fence, the SWE expert's training
  trajectories use v1's `bash`, and with the v2 fence the merged model's actions were unparseable
  (`RepeatedFormatError` on every turn). Applied to every model; runs started with the v2 fence were
  discarded (kept under `DISCARDED-mswea-*`, not reported).
- **ASI-Bench budget: 1,800 s per instance** (asibench default 10,800 s), the same for every model and
  fixed before any ASI-Bench result; at the default, about half the instances ran to the cap and one
  model's 236 runs would need ~40 GPU-hours on the available hardware.
- **Terminal-Bench seeds.** The exam asks for 5 runs per model. Measured throughput on the available
  GPUs is ~5-9 finished trials per hour per GPU (the 9B model thinks at length every turn), so one 89-task
  seed takes 10-18 GPU-hours. Seeds are run paired across models (seed 1 for base, term and merged first,
  then seed 2, ...) and the number of completed seeds per model is reported; incomplete seeds are
  reported with their task count and never mixed into the mean.
- **Per-call LLM timeout 600 s** (LiteLLM default inside Terminus-2), identical for every model; a call
  that exceeds it ends the trial (`llm_call_timeout`, counted as a failure and reported separately).
- **Concurrency** is set per GPU so that the sessions' KV working set fits the cache (it does not change
  any per-task setting): oversubscription had dropped the prefix-cache hit rate to ~16% and wasted most
  GPU time on re-prefill.

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
