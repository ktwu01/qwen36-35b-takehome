# Failure analysis of the merged model

Status: preliminary (runs in progress at 2026-10-19 18:40 CDT; Terminal-Bench seed 2 and 3 are partial).
Evidence: `analysis/failure_evidence.jsonl` (built by `scripts/failure_cases.py` from the raw runs:
exception, agent exit status, last agent turns, patch size, verifier tail for TB2 and SWE-Pro; score,
agent exit status and assistant turns for ASI-Bench; finish reason, output tokens and the judge's
extracted answer for HLE; for every case whether base or the matching expert solved the same task).
Cases are sampled uniformly at random (seed 0, 40 per benchmark) from the merged model's failed tasks;
41 of them are attributed below, each to one primary category, by reading its evidence. Aggregate
numbers in "Patterns" are over all tasks, not only the sample.

## Categories

The exam's seven categories. "Solved by" (base / expert) is reported for every case; it alone does
not make a case a merge-induced regression.

| Category | Definition used |
|---|---|
| Localization error | the agent edited or investigated the wrong file/function/component |
| Patch error | right place, wrong or incomplete change (logic, edge cases, build breaks, too slow) |
| Test-understanding error | misread the requirement or what the tests check; declared done with the requirement unmet |
| Merge-induced regression | base or the matching expert solved the task, the merged model did not, and the evidence shows a behavior base or the expert does not have (e.g. emitting an action format the harness rejects where base uses the accepted one) or, for single-turn HLE, a wrong answer where both reference models were right on that sample |
| Environment / tool failure | container, exec-server or sandbox failure (`env_error` status or a harness exception in the agent loop) |
| Timeout | agent time budget (TB2 3 h, SWE-Pro 50 min, ASI 1,800 s) or the HLE 32,768-token cap reached without an answer |
| Other | anything else, stated per case (action-format errors the harness rejects when no reference model solved the task; reasoning errors on HLE) |

## Cases

Run: TB2 seed; SWE-Pro and ASI-Bench have one run; HLE run1.

| # | Benchmark | Task | Run | Solved by | Category | Evidence (from the trajectory / verifier) |
|---|---|---|---|---|---|---|
| 1 | TB2 | large-scale-text-editing | s1 | none | Timeout | AgentTimeoutError at 3 h; last turns repeat "persistent JSON formatting issues ... try a completely different approach"; 3 of 5 tests fail |
| 2 | TB2 | code-from-image | s2 | base/term | Timeout | AgentTimeoutError; still trying salt/XOR variants to reproduce hash `bee26a`; output file never written (test_output_file_exists fails) |
| 3 | TB2 | fix-git | s1 | base and term | Localization error | restored about.md/default.html from `patch_files` and committed, but tests raise FileNotFoundError for both files at the expected paths |
| 4 | TB2 | pypi-server | s2 | base/term | Test-understanding error | declared done after `pip install --index-url http://localhost:8080/simple vectorops==0.1.0` succeeded in its shell; test_api fails with ModuleNotFoundError |
| 5 | TB2 | headless-terminal | s2 | base | Test-understanding error | "terminal output appears truncated, but the core functionality has been demonstrated" then marked complete; 6 of 7 tests fail with AttributeError (required methods missing) |
| 6 | TB2 | fix-code-vulnerability | s1 | base | Patch error | reported CWE-116 (test expects cwe-93) and "file corruption prevented full implementation"; hkey tests fail inside /app/bottle.py |
| 7 | TB2 | db-wal-recovery | s1 | none | Test-understanding error | "reconstructed all 11 records" by extrapolating an apple=100, banana=200 pattern instead of decrypting the WAL; test_wal_was_decrypted fails |
| 8 | TB2 | overfull-hbox | s1 | none | Test-understanding error | document compiles without overfull hbox, but input.tex was changed beyond the allowed synonym substitutions (test_input_file_matches fails) |
| 9 | TB2 | query-optimize | s2 | none | Patch error | CTE/window-function rewrite returns the same rows but test_compare_golden_vs_solution_runtime fails (slower than the reference) |
| 10 | TB2 | make-mips-interpreter | s1 | none | Patch error | interpreter prints "Doom MIPS execution complete"; verifier's VM run times out and /tmp/frame.bmp is never written |
| 11 | TB2 | distribution-search | s1 | none | Test-understanding error | declared both KL = 10.0 "mathematically impossible", saved KL1 = 9.71, KL2 = 1.90 and marked complete |
| 12 | TB2 | build-pov-ray | s3 | none | Test-understanding error | "unable to find POV-Ray 2.2 source after 50+ attempts", declared success with another build; test_povray_version fails |
| 13 | TB2 | fix-ocaml-gc | s1 | none | Timeout | AgentTimeoutError; last turns are Terminus-2's "Technical difficulties. Please continue with the task." after unparseable replies |
| 14 | TB2 | pypi-server | s1 | base/term | Environment / tool failure | RemoteProtocolError from the in-container exec server while the agent was freeing port 8080 (env_error, counted as a failure) |
| 15 | TB2 | regex-chess | s2 | none | Other (tool use) | "persistent issues with complex Python commands and escaping"; wrote a minimal re.json and stopped; 3 of 4 game tests fail |
| 16 | SWE-Pro | gravitational/teleport-f432a71a | - | none | Other (action format) | RepeatedFormatError after 8 calls, empty patch: replies had `content=null`, the bash fence was in `reasoning_content` |
| 17 | SWE-Pro | flipt-io/flipt-524f2773 | - | none | Other (action format) | RepeatedFormatError after 28 calls, empty patch, same mechanism |
| 18 | SWE-Pro | NodeBB/NodeBB-8168c6c4 | - | none | Other (action format) | RepeatedFormatError after 13 calls, empty patch, same mechanism |
| 19 | SWE-Pro | gravitational/teleport-65438e6e | - | base | Environment / tool failure | `sed -i` edit returned "Singularity environment not started" (returncode -1); time budget hit with an empty patch |
| 20 | SWE-Pro | internetarchive/openlibrary-b67138b3 | - | none | Environment / tool failure | same "Singularity environment not started" exception on a heredoc command; time budget |
| 21 | SWE-Pro | ansible/ansible-e9e60012 | - | base | Patch error | submitted a 31 KB rewrite of the WinRM kinit path; test_kinit_success_subprocess variants fail |
| 22 | SWE-Pro | ansible/ansible-d62496fe | - | base | Timeout | 30 calls spent on DEBUG prints inside human_to_bytes; time budget (50 min) reached before submitting |
| 23 | SWE-Pro | future-architect/vuls-e1fab805 | - | none | Patch error | submitted after 71 calls; 2 of 4 required tests fail (Test_debian_parseInstalledPackages, Test_detect) |
| 24 | SWE-Pro | flipt-io/flipt-c188284f | - | none | Timeout | after 27 calls still running `find /app/internal ... -path "*/oci/*"` to locate auth code; 0 of 6 required tests |
| 25 | SWE-Pro | flipt-io/flipt-cd18e54a | - | none | Patch error | inserted `"time"` above the `import (` block of internal/config/config.go, breaking the package; 0 required tests pass |
| 26 | SWE-Pro | flipt-io/flipt-3d5a345f | - | none | Patch error | last build: `"go.opentelemetry.io/otel/propagation" imported and not used`; 1 of 5 required tests pass |
| 27 | SWE-Pro | internetarchive/openlibrary-a48fd6ba | - | none | Localization error | searching its own `/app/code_backup.py` copy for `work_object` instead of the module under test; 75 calls, time budget |
| 28 | SWE-Pro | protonmail/webclients-5f0745dd | - | none | Test-understanding error | submitted after 24 calls claiming all PR requirements met; all 3 Bitcoin.test.tsx tests fail, including "should render" |
| 29 | ASI | earth_science.farquhar_ballberry_photosynthesis b1 | seed31415 | base 40.2 | Merge-induced regression | 0 accepted turns, RepeatedFormatError after 10.5 s: each reply is a valid ```` ```bash ```` block, rejected because the agent's model class expects native tool calls, which base emits |
| 30 | ASI | physics.ks_2d_anisotropic b3 | seed31415 | base 22.2 | Merge-induced regression | same: 3 rejected replies (`THOUGHT: ... ```bash ls -la````), RepeatedFormatError after 12.8 s, no files written |
| 31 | ASI | biology.calcium_ip3_dyk_oscillations b2 | seed31415 | base 21.0 | Merge-induced regression | same mechanism, 0 accepted turns, 14.6 s |
| 32 | ASI | biology.protein_folding_go_model b2 | seed31415 | base 13.9 | Merge-induced regression | same mechanism, 0 accepted turns, 48.5 s |
| 33 | ASI | math.sigma_linear_diophantine b4 | seed31415 | base 11.0 | Merge-induced regression | same mechanism, 0 accepted turns, 12.8 s |
| 34 | ASI | computer_science.hmc_neal_funnel b1 | seed31415 | base 1.0 | Timeout | 99 assistant turns, killed at 1,838 s (1,800 s budget), no scored output |
| 35 | HLE | 67199159b976203dd1bab7c3 | run1 | none | Timeout | finish_reason=length at 32,768 tokens, no final answer |
| 36 | HLE | 671ee933019b32e00d827382 | run1 | none | Other (empty answer) | finish_reason=stop after 29,580 tokens with an empty answer field (all text in the reasoning channel) |
| 37 | HLE | 66b91693d86bff9a12fc1f99 | run1 | none | Other (degenerate output) | 28,644 tokens; final answer "IT IS TO THE GREAT OF THE GREAT" (expected a decoded sentence) |
| 38 | HLE | 670df2a72f80209631f57fc2 | run1 | none | Other (reasoning error) | answered 2^49 = 562949953421312 for the 50th segmented number; correct 132 |
| 39 | HLE | 668825f80a642802bdfeadfa | run1 | base and sci | Merge-induced regression | answered A (Egalitarian Dominance); correct D, which base and sci both gave |
| 40 | HLE | 6731deb93e52c2db8f318e74 | run1 | base | Merge-induced regression | answered 625; correct 729 |
| 41 | HLE | 6724a047d917564737255345 | run1 | sci | Merge-induced regression | answered F (holds for any max degree); correct B (d = 1) |

## Counts (the 41 attributed cases)

| Category | TB2 | SWE-Pro | ASI | HLE | Total |
|---|---|---|---|---|---|
| Localization error | 1 | 1 | 0 | 0 | 2 |
| Patch error | 3 | 4 | 0 | 0 | 7 |
| Test-understanding error | 6 | 1 | 0 | 0 | 7 |
| Merge-induced regression | 0 | 0 | 5 | 3 | 8 |
| Environment / tool failure | 1 | 2 | 0 | 0 | 3 |
| Timeout | 3 | 2 | 1 | 1 | 7 |
| Other | 1 | 3 | 0 | 3 | 7 |
| Total | 15 | 13 | 6 | 7 | 41 |

HLE single-sample merge-induced cases (39-41) are weak evidence: sampling is at temperature 1.0 and,
over all 2,500 questions, merged lost 113 base-correct questions and gained 140.

## Patterns (all tasks)

1. **ASI-Bench: the agent ran in tool-calling mode and the merged model emits text actions.**
   `scripts/run_asi.sh` builds the agent config from mini-swe-agent 2.4.6's builtin
   `mini_textbased.yaml`, which (unlike the SWE-Pro task config) does not set
   `model_class: litellm_textbased`; mini-swe-agent then uses its default tool-calling model class
   while the prompt templates ask for a ```` ```bash ```` block. The config is identical for every model,
   but it is not neutral: base answers with native tool calls (192 of 240 runs contain at least one;
   48 runs end in RepeatedFormatError), while sci and merged follow the prompt's text format (tool
   calls in 43 and 40 runs; RepeatedFormatError in 197 and 206 runs). Of merged's 646 rejected replies,
   641 contain a well-formed ```` ```bash ```` block. 75 instances where base scored above 0 score 0 for
   merged. This, not science ability, is most of the ASI drop (base 5.78, merged 0.51, sci 0.48).
   Disclosed in README section 5; not re-run.
2. **SWE-Pro: reasoning-parser interaction.** Every format-rejected reply on SWE-Pro (base 5, swe 40,
   merged 52 replies; RepeatedFormatError exits base 1/51, swe 9/51, merged 7/51) has `content=null`
   with the bash fence inside `reasoning_content`: the model answered without a closing think tag and
   vLLM's `qwen3` reasoning parser filed the whole reply as reasoning, so the agent saw no action.
   Serving and agent config were frozen and identical for all models. Merged: 0/51 resolved; agent
   exits Submitted 23, time budget 21, RepeatedFormatError 7; 7 empty patches. Base resolves 3
   (2 ansible, 1 teleport); merged loses all 3 (cases 19, 21, 22).
3. **TB2: fewer timeouts, more early "task complete".** Seed 1: merged 11/89, base 17/89, term 8/89.
   AgentTimeoutError in seed 1: base 25, term 21, merged 12. The merged model ends trials sooner, and
   in the sample 6 of 15 TB2 failures are a declared completion with the requirement unmet (cases 4, 5,
   7, 8, 11, 12). Merged loses 10 base-solved seed-1 tasks and solves 3 that neither base nor term
   solved (constraints-scheduling, log-summary-date-ranges, portfolio-optimization).
4. **Environment failures are a small, model-independent share.** Merged TB2 env errors over seeds 1-2:
   ConnectError 3, RuntimeError 4 (in-container tmux server gone after the agent sent keys such as ^D),
   MemoryLimitExceededError 2, RemoteProtocolError 1; base seed 1 has 4 (ConnectError 2, RuntimeError 1, MemoryLimitExceededError 1). SWE-Pro
   "Singularity environment not started" occurs mid-trajectory for both merged and base.
5. **HLE: shorter reasoning is the merged model's gain.** Base hits the 32,768-token cap on 1,374 of
   2,500 questions (55%), sci 600, merged 427; unanswered questions score 0. Merged 8.16% vs base 7.08%
   (+1.08 pp, 95% CI [-0.16, 2.32]).

## Next experiments

1. **One action protocol across training data, harness and serving.** Re-run ASI-Bench for base, sci
   and merged with `model_class: litellm_textbased` (the protocol of the SWE-Pro config and of the swe
   expert's training data), and measure the RepeatedFormatError rate first on the ASI dev instances.
   Pattern 1 predicts most of the merged ASI gap closes; if it does not, the remainder is a science
   ability effect.
2. **Fix the parser interaction instead of the model.** Serve with a reasoning parser that leaves a
   reply without think tags in `content`, or let the agent parse the action from `reasoning_content`
   when `content` is empty; re-run SWE-Pro HARD-51. This removes the 7 merged and 9 swe RepeatedFormatError
   exits (pattern 2) and separates format failures from coding failures.
3. **Select the merge on agent behavior, not dev loss.** The TIES configuration was chosen on dev-set
   loss, which did not predict agent outcomes: merged is below base on SWE-Pro, TB2 and ASI. Score the
   ablation grid (TIES/DARE densities, lambda 0.5-1.0, and keeping base weights for attention or
   embeddings) on 20-task dev subsets of SWE-Pro and TB2 plus the format-error rate, and pick on that.
4. **Expert data quality for TB2.** The term expert itself is below base (8/89 vs 17/89 on seed 1),
   so its data hurts before any merge. Filter OpenThoughts-Agent trajectories to verified successes,
   train with the TB2 context actually reached (16K training context vs much longer TB2 sessions), and
   add examples that verify requirements before declaring completion (pattern 3).
5. **HLE budget.** Re-measure base and merged with a larger token cap on a fixed subset to separate
   accuracy from the length effect in pattern 5.
