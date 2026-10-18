# Failure analysis of the merged model

Evidence: `analysis/failure_evidence.jsonl` (built by `scripts/failure_cases.py` from the raw runs:
exception, agent exit status, last agent turns, patch size, verifier tail, and whether base or the
matching expert solved the same task). Cases are sampled uniformly at random (seed 0) from the merged
model's failed tasks per benchmark; each is attributed to one primary category by reading its evidence.

## Categories

| Category | Definition used |
|---|---|
| Localization error | the agent edited or investigated the wrong file/function/component |
| Patch error | right place, wrong or incomplete change (logic, edge cases, regressions in other tests) |
| Test-understanding error | misread the requirement/spec or what the hidden tests check; declared done early |
| Merge-induced regression | base or the matching expert solved the task, the merged model did not, and the trajectory shows a behavior the expert does not have (format drift, protocol confusion, loss of domain knowledge) |
| Action-format error | the harness could not parse the action (`RepeatedFormatError`, malformed Terminus-2 JSON) |
| Environment / tool failure | container, exec-server, network or verifier infrastructure failure (`env_error` status) |
| Timeout | agent hit its time budget (`agent_timeout`) or one LLM call exceeded 600 s (`llm_call_timeout`) |
| Other | anything else (stated explicitly) |

## Cases

(filled from the evidence file once the runs finish)

## Counts and patterns

(pending)

## Next experiments

(pending; written from the observed failure modes)
