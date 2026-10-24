# Log index

All paths are relative to the repository root. Text logs are plain UTF-8 (`less`, `grep`); `.json` /
`.jsonl` files are read with `python -m json.tool` or `jq`; `.csv` with any spreadsheet or pandas.

## Training (one directory per expert: `logs/train-<swe|term|sci|union>/`)

| File | Format | Content |
|---|---|---|
| `train.log` | text | trainer stdout/stderr of every session (resumed jobs appended) |
| `train_seismic18.log` | text | union only: the reported single-GPU run (its `train.log` is the failed 3-GPU DDP attempt) |
| `metrics.jsonl` | JSON lines | per-step train loss, learning rate, grad norm, elapsed time, peak GPU memory; `eval_loss` rows = dev loss (source of `curves/`) |
| `resources.csv` | CSV | per session: GPU model, wall time, peak memory, tokens, tokens/s, adapter size (README section 3) |
| `run_meta.json` | JSON | resolved training config snapshot (incl. seed), host, selected checkpoint |

Dependency locks: `requirements*.lock` (repository root), `logs/train_reqs.txt`, `logs/pg_venv_freeze.txt`
(eval venv), `logs/mkvenvs.sh` + `logs/mkvenvs.log` (how the venvs were built).

## Evaluation

| Files | Content |
|---|---|
| `logs/tb2/<model>-<host>.log` | Terminal-Bench runner output per model and serving endpoint (Harbor + Terminus-2) |
| `logs/sbp_<model>*.log` | SWE-bench Pro agent runs and `patch_replay` regrades (`sbp_swe_p2..p5`: swe expert parts) |
| `logs/asi_<model>.log`, `logs/asi_score_<model>.log` | ASI-Bench runs and official local scoring |
| `logs/hle_<model>.log`, `logs/hle_rejudge.log` | HLE generation and o3-mini judging (re-judge of HTTP 429 failures) |
| `logs/vllm_<model>_<host>.log` | vLLM serving logs (start/end times feed `logs/eval_lanes.json`) |
| `logs/*oracle*.log`, `logs/sbp_check*.log`, `logs/sbp_gate*.log` | environment validation (oracle runs) |
| `logs/lane_*.log`, `logs/handoff_progress.log`, `logs/report_progress.log` | lane controllers and the time-stamped operations/decision log, including every incident (e.g. the seismic18 reboot) |

Errors and timeouts: per task in `results/<group>/<benchmark>/<model>/` and `results/failures/<model>_<benchmark>.csv`
(status column: `agent_timeout`, `llm_call_timeout`, `env_error`, `no_result`); attributed merged-model
failures in `analysis/failures.md` (evidence in `analysis/failure_evidence.jsonl`).

## Resources and cost

`logs/eval_lanes.json` (serving GPU-hours per lane), `logs/hpc_sacct.txt` (HPC SU estimate and how it was
built), `logs/cost.json` (output of `scripts/cost_report.py`), `logs/train-*/resources.csv` (training).

## Raw per-trial records (compressed, GitHub Release)

Every trial directory of every evaluation run (Terminal-Bench `tb2-jobs/`, SWE-bench Pro `sbp-jobs/`,
ASI-Bench `asi-runs/`): `result.json`, `config.json`, agent trajectories, verifier output, patches and
per-trial logs (text files up to 20 MB each; container images and build artifacts excluded).

- Asset: `eval_raw_trials.tar.gz` in release `eval-raw-v1` of this repository
- sha256: see `logs/eval_raw_trials.tar.gz.sha256`
- View: `tar -xzf eval_raw_trials.tar.gz`, then e.g. `jq . tb2-jobs/<job>/<trial>/result.json`

Regenerate all result tables from these records:
`python3 scripts/summarize.py --base <extracted directory>` (writes `results/`, `results/summary.csv`,
`analysis/uncertainty.json`), then `python3 scripts/results_table.py` and `python3 scripts/make_report.py`.
HLE per-question predictions and judgments are already in `results/<group>/hle/<model>/run1/`.
