"""Loss curves per expert and merge-ablation dev-loss comparison, from raw metrics.

    python scripts/plot_curves.py --runs outputs --dev results/dev

Inputs: logs/train-<expert>/metrics.jsonl (written every step by takehome.sft) and
results/dev/<config>.json (takehome.dev_loss). Writes curves/source/*.csv (the plotted data),
curves/expert_loss.png and curves/merge_ablation.png. Train loss is plotted raw (thin) plus a
10-step trailing mean; dev loss is every 25 steps on the first 100 dev examples.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (light)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
EXPERTS = ["swe", "term", "sci", "union"]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def load_metrics(path):
    train, dev = [], []
    for line in open(path):
        r = json.loads(line)
        if "loss" in r and "eval_loss" not in r:
            train.append((r["step"], r["loss"]))
        if "eval_loss" in r:
            dev.append((r["step"], r["eval_loss"]))
    return sorted(dict(train).items()), sorted(dict(dev).items())


def expert_curves(runs, out):
    def mpath(e):  # logs/train-<e>/metrics.jsonl (repo layout) or outputs/<e>/metrics.jsonl
        return next((p for p in (runs / f"train-{e}" / "metrics.jsonl", runs / e / "metrics.jsonl") if p.exists()), None)
    avail = [e for e in EXPERTS if mpath(e)]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.6 * len(avail), 3.0), facecolor="#fcfcfb")
    axes = [axes] if len(avail) == 1 else axes
    for ax, e, color in zip(axes, avail, SERIES):
        train, dev = load_metrics(mpath(e))
        with open(out / "source" / f"loss_{e}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "step", "loss"])
            w.writerows([("train", s, l) for s, l in train] + [("dev", s, l) for s, l in dev])
        steps, losses = zip(*train)
        roll = [sum(losses[max(0, i - 9):i + 1]) / len(losses[max(0, i - 9):i + 1]) for i in range(len(losses))]
        ax.plot(steps, losses, color=color, alpha=0.25, linewidth=1)
        ax.plot(steps, roll, color=color, linewidth=2, label="train (10-step mean)")
        if dev:
            ds, dl = zip(*dev)
            ax.plot(ds, dl, color=INK, linewidth=1.5, marker="o", markersize=4, label="dev")
            ax.annotate(f"dev {dl[-1]:.3f}", (ds[-1], dl[-1]), xytext=(4, 6), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_title(f"{e} expert" if e != "union" else "union control", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("optimizer step (final checkpoint = last step)", fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_ylabel("token cross-entropy (assistant tokens)", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.suptitle("QLoRA training on Qwen3.5-9B: train and dev loss", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "expert_loss.png", dpi=160)


def merge_ablation(dev_dir, out):
    rows = []
    for p in sorted(Path(dev_dir).glob("*.json")):
        if p.name.endswith((".recipe.json", ".merge_stats.json")):
            continue
        r = json.loads(p.read_text())
        rows.append([r["name"]] + [r["domains"][d]["token_loss"] for d in ("swe", "term", "sci")] + [r["mean_token_loss"]])
    if not rows:
        return
    rows.sort(key=lambda r: r[-1])
    with open(out / "source" / "merge_ablation_dev_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "swe_dev_loss", "term_dev_loss", "sci_dev_loss", "mean_dev_loss"])
        w.writerows(rows)
    fig, axes = plt.subplots(1, 4, figsize=(13, 0.32 * len(rows) + 1.4), sharey=True, facecolor="#fcfcfb")
    names = [r[0] for r in rows]
    for j, (ax, title) in enumerate(zip(axes, ["swe dev", "term dev", "sci dev", "mean of 3"])):
        vals = [r[j + 1] for r in rows]
        ax.hlines(range(len(rows)), min(vals) * 0.98, vals, color=GRID, linewidth=1)
        ax.plot(vals, range(len(rows)), "o", color=SERIES[j] if j < 3 else INK, markersize=6)
        for y, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(5, -3), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("dev token loss (lower is better)", fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_yticks(range(len(rows)), names, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Merge ablation on independent dev sets (sorted by mean)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "merge_ablation.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="logs")
    ap.add_argument("--dev", default="results/dev")
    ap.add_argument("--out", default="curves")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "source").mkdir(parents=True, exist_ok=True)
    expert_curves(Path(a.runs), out)
    merge_ablation(a.dev, out)


if __name__ == "__main__":
    main()
