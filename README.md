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
  vs 0.1706 (still improving); swe expert at 66% (checkpoint 75) vs final: swe 0.2094 vs 0.2083 (nearly
  converged). The planned swe checkpoint 50 had been rotated out by `save_total_limit`, so checkpoint 75
  was used.
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
  environment. SWE-Pro: oracle 50/51, nop 0/51. Terminal-Bench: the oracle passes 59 of 89 tasks
  (`tb2-jobs/oracle-full-env4`, 57, plus 2 more in a 3 h re-run of the 32 failures); 20 of the 30
  failures are the reference solution itself running past the agent budget. Oracle failure does not
  make a task unsolvable here (base, term and merged together solve 7 of those 30 tasks on seed 1), so
  no task is dropped: all 89 stay in every denominator. On the 59 oracle-passing tasks, seed 1: base
  20.34%, term 10.17%, merged 15.25%.
- **CPU limit not enforced.** The apptainer backend has no cgroups; the 48 GB memory cap is enforced by
  Harbor's watchdog, 32 CPUs is requested but not enforced.
- **SWE-Pro agent phase not network-isolated.** The locked protocol's `no-network` agent phase cannot be
  enforced by this backend; `configs/eval/swe_pro_task_modifications.json` records the one-line change
  per task. Trajectories are audited for network use (`scripts/audit_network_use.py`,
  `analysis/network_audit.json`): network commands (`go get`, `curl`, `pip install`) appear in 6 of 51
  base, 1 of 51 swe and 0 of 51 merged trajectories; none of base's 3 resolved tasks is among them.
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
  GPU time on re-prefill. HLE generation: at most 12 requests in flight per endpoint while that model's
  ASI-Bench run shared the GPU, 24 after it, the same rule for every model; questions were routed to a
  model's endpoints by a hash of the question id.
- **Endpoint-unavailable rule** (fixed 2026-10-18 14:20 CDT, same for every model): a Terminal-Bench or
  SWE-Pro trial that ends with `InternalServerError` (the model endpoint was unavailable: vLLM restart or
  the end of a HPC serving window) is treated like an operator cancellation: not scored, the task is
  re-run in a continuation part (`run_tb2.py --part N`, `PART=N scripts/run_swepro.sh`). Trials cancelled
  by stopping a runner (no `finished_at`) are handled the same way. `ConnectError` / `RuntimeError`
  (task-container exec-server failures) stay environment failures in the denominator. When the rule was
  introduced it affected only 7 merged Terminal-Bench seed-1 trials. Every task is scored from exactly
  one finished trial per seed; no finished task was re-run.
- **ASI-Bench agent ran in mini-swe-agent's tool-calling mode.** `scripts/run_asi.sh` builds the
  config from mini-swe-agent 2.4.6's builtin `mini_textbased.yaml`, which (unlike the SWE-Pro task
  config) does not set `model_class: litellm_textbased`, so the default model class, which accepts only
  native tool calls (vLLM `qwen3_coder` tool parser), was used while the prompt templates ask for a
  ```` ```bash ```` block. This was found while writing the failure analysis, after all ASI runs; it is
  identical for every model but not neutral: base mostly answers with native tool calls, sci and merged
  answer with the requested text block and are rejected (`RepeatedFormatError` in 48 base, 197 sci and
  206 merged runs of 240; `analysis/failures.md`, pattern 1). Not re-run; reported as measured.
- **SWE-Pro and the `qwen3` reasoning parser.** When a model replies without closing its think block,
  vLLM's `qwen3` reasoning parser returns the whole reply as `reasoning_content` with `content=null`, so
  the agent sees no action and eventually exits with `RepeatedFormatError` (base 1, swe 9, merged 7 of
  51 trials). Serving and agent configs were frozen and identical; counted as failures.
- **ASI-Bench judge and scorer.** The task contracts name `openai/gpt-5.5` as judge with temperature
  0.0, which the OpenAI API rejects for that model ("only the default (1) value is supported");
  `LITELLM_DROP_PARAMS` drops it, so the judge runs at the API default for every model.
  `computer_science.euclidean_tsp_tour_optimization`'s validity gate needs Docker (not available without
  root): those instances are scorer-invalid for every model; the official mean excludes them and the
  all-instances mean counts them 0. asibench's `run` writes outputs only; scoring is
  `scripts/score_asi.sh` (asibench's local scorer) on the 236 locked instances.
- **ASI-Bench contention period.** From 09:41 to 15:17 CDT on 2026-10-18 the A100 that served base was
  also loaded by orphaned agent processes (asibench's timeout killed the sandbox but not mini-swe-agent;
  fixed by `scripts/reap_asi_orphans.sh`, no score changes) and by 48 HLE requests. Timeouts were not
  re-run (re-running only failures would be a biased extra draw). Instances stopped at the 1,800 s
  budget, by start time (start = result-file time minus execution time): base 20/38 before 15:17 and
  99/198 after; sci 9/63 and 21/173; merged 10/222 and 1/14.
- **HLE judge errors.** 71 sci and 139 merged answers (0 base) were not judged on the first pass because
  the OpenAI API returned HTTP 429; they were judged again with the same judge and prompt
  (`eval_hle judge --retry-judge-errors`; the failed records are kept in `judged.errors.jsonl`). No model
  output was regenerated. Answers with no final answer (32,768-token cap) score 0 and stay in the
  denominator.
- **Run placement and interruptions** (same serving flags and quantization everywhere; host per trial
  in the job directories and logs):
  - seismic18 stopped responding after 02:46 CDT on 2026-10-19 and was rebooted at 08:03; nothing ran on
    it from ~03:00 to 11:20. Its in-flight trials have no result and were re-run as continuation parts.
  - SWE-Pro: base ran on seismic18 (RTX PRO 4000), merged on seismic17 (RTX A4500), the swe expert in
    4 disjoint task parts on 3 HPC A100s and seismic18 (`SBP_SHARD`, commit 3ab96db), same agent
    config and budget.
  - Terminal-Bench: base seed 1 on seismic18 and a HPC A100 slot; base seed 2 on seismic18, HPC dev windows
    and, for its last 15 tasks, seismic15; base seed 3 split between seismic15 and dev windows. Merged seed 1
    on seismic17 and dev windows, except 2 trials (caffe-cifar-10, feal-differential-cryptanalysis,
    3 h budget) moved to the A100-small slot because a dev window (~1 h 45 min of serving) cut them every
    time; merged seeds 2 and 3 in disjoint shards on seismic17 and seismic18. Re-splitting a stalled
    shard cancelled its in-flight trials (unscored, re-run).
  - term expert seed 2 was stopped at 4/89 finished on 2026-10-19 15:19 when seismic15 was reassigned to
    base seed 2, so term has one complete seed.
- **Terminal-Bench seeds < 5.** No model has 5 complete seeds (section 6 lists complete and partial
  seeds per model); the mean is over complete seeds only and the seed std is not available with one.

## 6. Results

<!-- results:begin -->
Status: **preliminary: runs in progress at 2026-10-19 17:51 CDT; Terminal-Bench seeds 2-3 are partial**. Generated by `scripts/results_table.py` from `results/summary.csv`,
`analysis/uncertainty.json` and the per-task CSVs; per-task results in `results/<group>/<benchmark>/<model>/`,
failed-task lists in `results/failures/`.

Primary metric, percent. Experts are evaluated on their target benchmarks only (n/e: not evaluated):
swe on SWE-Pro, term on Terminal-Bench, sci on HLE and ASI-Bench. Terminal-Bench: mean +- std over
complete seeds (all 89 tasks finished); partial seeds are listed below and are not in the mean.

| Benchmark | Base | swe | term | sci | Merged | Merged - Base, pp (95% CI) |
|---|---|---|---|---|---|---|
| SWE-bench Pro v2-hard (HARD-51) | 5.88 | 0.00 | n/e | n/e | 0.00 | -5.88 [-13.73, +0.00] (n=51) |
| Terminal-Bench 2.0 (89 tasks) | 19.10 (1 seed) | n/e | 8.99 (1 seed) | n/e | 12.36 (1 seed) | -6.74 [-14.61, +1.12] (n=89) |
| HLE (2,500) | 7.08 | n/e | n/e | 7.12 | 8.16 | +1.08 [-0.16, +2.32] (n=2500) |
| ASI-Bench (59 x 4 = 236) | 5.78 | n/e | n/e | 0.48 | 0.51 | -5.13 [-6.70, -3.69] (n=236) |
| Macro average (4 benchmarks, equal weight) | 9.46 | | | | 5.26 | -4.17 [-6.92, -1.61] |

Merged - Base is a paired bootstrap over the same tasks (10,000 resamples); Terminal-Bench uses complete
seeds only; for ASI-Bench the paired difference is over all 236 instances with scorer-invalid instances
scored 0 (the all-instances mean below), so it can differ slightly from the difference of the official means.
The macro CI resamples tasks within each benchmark jointly. The exam's target is +3 pp (excellent +5 pp).

**Terminal-Bench seeds** (success rate, percent; partial = tasks finished so far, success on those):

| Model | Complete seeds | Partial seeds |
|---|---|---|
| base | seed1: 19.10 | seed2: 74/89 finished, 28.4, seed3: 9/89 finished, 22.2 |
| term | seed1: 8.99 | seed2: 4/89 finished, 0.0 |
| merged | seed1: 12.36 | seed2: 77/89 finished, 11.7, seed3: 12/89 finished, 16.7 |

The exam asks for 5 seeds per model; seeds 2-5 could not be completed in the compute available
(section 5). The seed std of a single complete seed is not defined and is not reported.

**Run outcomes and failure rate** (every finished trial / question / instance; failures stay in the
denominator and score 0). Failure rate = share not `ok`.

| Model | Benchmark | n | ok | agent_timeout | llm_call_timeout | env_error | no_result | failure rate |
|---|---|---|---|---|---|---|---|---|
| base | swe_pro | 51 | 51 | 0 | 0 | 0 | 0 | 0.0% |
| base | tb2 | 172 | 126 | 36 | 1 | 9 | 0 | 26.7% |
| base | hle | 2500 | 1220 | 0 | 0 | 0 | 1280 | 51.2% |
| base | asi | 236 | 230 | 0 | 0 | 0 | 6 | 2.5% |
| swe | swe_pro | 51 | 51 | 0 | 0 | 0 | 0 | 0.0% |
| term | tb2 | 93 | 69 | 21 | 2 | 1 | 0 | 25.8% |
| sci | hle | 2500 | 1916 | 0 | 0 | 0 | 584 | 23.4% |
| sci | asi | 236 | 234 | 0 | 0 | 0 | 2 | 0.8% |
| merged | swe_pro | 51 | 51 | 0 | 0 | 0 | 0 | 0.0% |
| merged | tb2 | 178 | 148 | 20 | 0 | 10 | 0 | 16.9% |
| merged | hle | 2500 | 2049 | 0 | 0 | 0 | 451 | 18.0% |
| merged | asi | 236 | 235 | 0 | 0 | 0 | 1 | 0.4% |

Status meaning: TB2 `agent_timeout` = 3 h agent budget, `llm_call_timeout` = one model call over 600 s,
`env_error` = container / exec-server failure; SWE-Pro rows are the patch_replay regrade (agent exits are
in `analysis/failures.md`); HLE `no_result` = no final answer (32,768-token cap reached, or empty answer);
ASI `no_result` = scorer-invalid instance (Docker-only validity gate, section 5).

**Secondary metrics**

- base: HLE text-only subset 6.49; ASI all-instances mean 5.63; per level b1 8.18, b2 5.49, b3 4.46, b4 4.40; scorer-invalid 6; agent stopped at the 1,800 s budget 119/236 (graded as is).
- sci: HLE text-only subset 6.86; ASI all-instances mean 0.47; per level b1 1.06, b2 0.39, b3 0.28, b4 0.16; scorer-invalid 2; agent stopped at the 1,800 s budget 30/236 (graded as is).
- merged: HLE text-only subset 8.39; ASI all-instances mean 0.50; per level b1 0.33, b2 1.15, b3 0.04, b4 0.50; scorer-invalid 1; agent stopped at the 1,800 s budget 11/236 (graded as is).

**Time and cost per task** (mean wall time; tokens are model tokens served by our vLLM, no API cost):

| Model | Benchmark | mean time | mean output tokens |
|---|---|---|---|
| base | swe_pro | 36.5 min / agent trial | |
| base | tb2 | 64.1 min / trial | 31,510 |
| base | hle | 715 s / question | 25,938 |
| base | asi | 1926 s / instance (incl. sandbox setup) | |
| swe | swe_pro | 6.7 min / agent trial | |
| term | tb2 | 77.9 min / trial | 53,022 |
| sci | hle | 323 s / question | 16,744 |
| sci | asi | 411 s / instance (incl. sandbox setup) | |
| merged | swe_pro | 29.9 min / agent trial | |
| merged | tb2 | 43.0 min / trial | 27,222 |
| merged | hle | 332 s / question | 15,618 |
| merged | asi | 215 s / instance (incl. sandbox setup) | |

Compute: HPC estimated 181.8 SU (the project allocation; A100 GPU-hours: train-experts 10.0, merge-ablation 3.0, serve-slot6 36.5, serve-slot7 24.0, serve-slot8 24.0, serve-dev-windows-x7 42.0), seismic serving 106.1 GPU-hours (`logs/eval_lanes.json`), cloud rental 0 USD. The HPC figure is an
estimate because sacct cannot be fetched from seismic (`logs/hpc_sacct.txt` states how it was built).
API cost: HLE judge o3-mini and ASI-Bench judge gpt-5.5 calls on the OpenAI API (usage not logged per call).
<!-- results:end -->

**Reading the results.** The merged model is below base on the macro average, with a 95% CI that
excludes 0, so the +3 pp target is not met. Per benchmark: SWE-Pro and Terminal-Bench regress (base
solves 3 SWE-Pro tasks the merged model does not; on Terminal-Bench seed 1 the merged model loses 10
base-solved tasks and gains 3), HLE improves slightly, mostly because the merged model stops within
the 32,768-token cap far more often than base, and ASI-Bench drops sharply. The ASI drop is mainly a
protocol mismatch, not lost science ability: the ASI agent ran in mini-swe-agent's tool-calling mode
(section 5), base answers with native tool calls, while sci and merged answer with the text `bash`
blocks the prompt asks for, which that mode rejects (`RepeatedFormatError` in 206 of 240 merged runs).
The experts themselves do not beat base on their own targets (swe 0/51 on SWE-Pro, term below base
on Terminal-Bench seed 1; sci within noise of base on HLE), so the merge had little expert gain to
keep. Failure cases and next experiments: [analysis/failures.md](analysis/failures.md).

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
- One training seed per expert. Repeat-run variance was planned on Terminal-Bench (5 seeds) but each
  model has only one complete seed, so the Terminal-Bench numbers are single-seed and no seed std is
  available; partial seeds are reported separately.
- ASI-Bench results reflect the tool-calling agent mode (section 5): they measure the models' ability
  to use that protocol as much as science ability, and they penalize the fine-tuned models, which
  follow the prompt's text action format.
- SWE-Pro and ASI-Bench format failures from the `qwen3` reasoning parser and the tool-call mode are
  part of the measured scores; a parser or harness change would change them.
- HLE accuracy depends strongly on the 32,768-token answer cap (base reaches it on 55% of questions).
- HPC SU usage is an estimate (sacct not reachable from the eval hosts); seismic GPU-hours are
  wall-clock serving time, not utilization.
- ASI-Bench runs at a 1,800 s budget (asibench default 10,800 s); about half of base's instances reach it.
