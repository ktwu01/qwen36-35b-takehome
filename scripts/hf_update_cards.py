"""Refresh the Evaluation section (and merge notes) of the four Hugging Face model cards.

    HF_HOME=/var/tmp/$USER/hf-cache python scripts/hf_update_cards.py --status "preliminary, ..."

Downloads each card (README.md), replaces everything between `## Evaluation` and the next `## ` heading
with results rendered from analysis/uncertainty.json, adds a `## Use in the merge` section to the expert
cards (recipe: configs/merge_final.json), uploads only README.md (weights untouched) and records the new
revision in models/hf_index.json.
"""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

NAMES = {"expert-swe": "swe", "expert-term": "term", "expert-sci": "sci", "merged": "merged"}
BENCH = {"swe_pro": "SWE-bench Pro v2-hard (HARD-51, resolved %)", "tb2": "Terminal-Bench 2.0 (89 tasks, success %)",
         "hle": "HLE (2,500 questions, accuracy %)", "asi": "ASI-Bench (236 instances, official mean %)"}
REPO = "https://github.com/ktwu01/qwen36-35b-takehome"


def results_md(model, unc, status):
    met, boot = unc["metrics"], unc["paired_bootstrap_merged_minus_base"]
    L = [f"Status: {status}. Experiment-specific protocol (apptainer containers, FP8 weights and KV cache, "
         f"thinking on, temperature 1.0); not official leaderboard numbers. Full tables, per-task results and "
         f"failure analysis: {REPO} (README section 6, analysis/failures.md).", "",
         "| Benchmark | This model | Base (Qwen3.5-9B) |" + (" Merged - Base, pp (95% CI) |" if model == "merged" else ""),
         "|---|---|---|" + ("---|" if model == "merged" else "")]
    for b, name in BENCH.items():
        t = met.get(f"{model}/{b}")
        if not t:
            continue
        base = met.get(f"base/{b}")

        def v(x):
            if x is None:
                return ""
            s = f"{x['primary']:.2f}"
            if b == "tb2":
                s += f" ({x['n_complete_seeds']} complete seed{'s' if x['n_complete_seeds'] != 1 else ''})"
            return s
        row = f"| {name} | {v(t)} | {v(base)} |"
        if model == "merged" and b in boot:
            d = boot[b]
            row += f" {d['diff_pp']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] |"
        L.append(row)
    if model == "merged" and "macro_merged_minus_base" in unc:
        m = unc["macro_merged_minus_base"]
        L += ["", f"Macro average over the four benchmarks, merged - base: {m['diff_pp']:+.2f} pp "
                  f"(95% CI [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}]); the experiment's +3 pp target is not met."]
    if model != "merged":
        L += ["", "Experts were evaluated on their target benchmarks only."]
    notes = {
        "merged": "ASI-Bench ran in mini-swe-agent's tool-calling mode; this model answers with text bash blocks, which "
                  "that mode rejects (RepeatedFormatError in 206 of 240 runs), so its ASI score mostly reflects the "
                  "protocol mismatch. Terminal-Bench has fewer than the planned 5 seeds.",
        "sci": "ASI-Bench ran in mini-swe-agent's tool-calling mode; this model answers with text bash blocks, which "
               "that mode rejects (RepeatedFormatError in 197 of 240 runs).",
        "swe": "9 of 51 SWE-Pro trials end in RepeatedFormatError: replies without a closing think tag are returned "
               "by vLLM's qwen3 reasoning parser as reasoning_content only.",
        "term": "Terminal-Bench has fewer than the planned 5 seeds.",
    }
    L += ["", notes[model]]
    return "\n".join(L) + "\n"


def merge_note(recipe):
    return ("## Use in the merge\n\nThis adapter's dense LoRA delta is one of the three task vectors of "
            "[ktwu01/qwen35-9b-takehome-merged](https://huggingface.co/ktwu01/qwen35-9b-takehome-merged), "
            "merged with TIES (recipe below, `configs/merge_final.json` in the code repo; implementation "
            "`src/takehome/merge.py`).\n\n```json\n" + json.dumps(recipe, indent=1) + "\n```\n\n")


def replace_section(card, heading, body):
    m = re.search(rf"^## {re.escape(heading)}\n.*?(?=^## |\Z)", card, flags=re.S | re.M)
    new = f"## {heading}\n\n{body}\n"
    return card[:m.start()] + new + card[m.end():] if m else card.rstrip() + "\n\n" + new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    a = ap.parse_args()
    unc = json.load(open("analysis/uncertainty.json"))
    recipe = json.load(open("configs/merge_final.json"))
    idx_path = Path("models/hf_index.json")
    idx = json.loads(idx_path.read_text())
    api = HfApi()
    for name, model in NAMES.items():
        repo = idx[name]["repo"]
        card = Path(hf_hub_download(repo, "README.md", revision=idx[name]["revision"])).read_text()
        card = replace_section(card, "Evaluation", results_md(model, unc, a.status))
        if name != "merged":
            card = re.sub(r"^## Use in the merge\n.*?(?=^## )", "", card, flags=re.S | re.M)
            card = card.replace("## Loading\n", merge_note(recipe) + "## Loading\n", 1)
        info = api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                               repo_type="model", commit_message="Model card: evaluation results")
        idx[name]["revision"] = info.oid
        print(name, info.oid)
    idx_path.write_text(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
