"""Pick the final merge configuration from the dev-set ablation and freeze it.

    python scripts/select_merge.py --dev results/dev        # writes configs/merge_final.json

Rule (fixed before looking at any benchmark result): among merges of all three experts, the lowest
mean of the three per-domain dev token losses; ties broken by the lower worst-domain loss. Only
independent dev sets are used. Single experts, checkpoints and 2-expert merges are controls and
are never selected.
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/dev")
    ap.add_argument("--grid", default="configs/merge_ablation.tsv")
    ap.add_argument("--out", default="configs/merge_final.json")
    a = ap.parse_args()
    grid = {r["name"]: r["merge_args"] for r in csv.DictReader(open(a.grid), delimiter="\t")}
    cands = []
    for name, args in grid.items():
        three = "outputs/swe outputs/term outputs/sci" in args and "--method single" not in args
        f = Path(a.dev) / f"{name}.json"
        if three and f.exists():
            r = json.loads(f.read_text())
            losses = {d: v["token_loss"] for d, v in r["domains"].items()}
            cands.append((r["mean_token_loss"], max(losses.values()), name, args, losses))
    if not cands:
        raise SystemExit("no 3-expert merge results yet")
    cands.sort()
    mean, worst, name, args, losses = cands[0]
    final = {"name": name, "merge_args": args, "dev_mean_token_loss": mean, "dev_token_loss": losses,
             "rule": "lowest mean dev token loss over swe/term/sci among 3-expert merges; tie -> lower worst domain",
             "ranking": [{"name": c[2], "mean": round(c[0], 5), "worst": round(c[1], 5)} for c in cands]}
    Path(a.out).write_text(json.dumps(final, indent=1))
    print(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
