"""Build the one-page report.pdf (摘要 - 过程 - 结果) from the result files.

    python scripts/make_report.py       # needs xelatex + ctex; writes report/report.tex and report.pdf

Text: report/text.yaml (abstract_zh 150-200 chars, process_zh, notes_zh). Numbers: results/summary.csv,
analysis/uncertainty.json, logs/cost.json. Every number in the table comes from those files.
"""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import yaml

BENCH = [("swe_pro", "SWE-bench Pro v2-hard"), ("tb2", "Terminal-Bench 2.0"), ("hle", "HLE"), ("asi", "ASI-Bench")]
MODELS = [("base", "Base"), ("swe", "swe 专家"), ("term", "term 专家"), ("sci", "sci 专家"), ("merged", "合版")]


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def main():
    text = yaml.safe_load(Path("report/text.yaml").read_text())
    summ = {(r["model"], r["benchmark"]): r for r in csv.DictReader(open("results/summary.csv"))}
    unc = json.loads(Path("analysis/uncertainty.json").read_text())
    cost = json.loads(Path("logs/cost.json").read_text()) if Path("logs/cost.json").exists() else {}
    boot = unc.get("paired_bootstrap_merged_minus_base", {})

    rows = []
    for b, bname in BENCH:
        cells = []
        for m, _ in MODELS:
            r = summ.get((m, b))
            if not r or r["primary_percent"] == "":
                cells.append("--")
                continue
            v = f"{float(r['primary_percent']):.1f}"
            if b == "tb2" and float(r.get("seed_std") or 0) > 0:
                v += f"$\\pm${float(r['seed_std']):.1f}"
            cells.append(v)
        d = boot.get(b)
        delta = f"{d['diff_pp']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d else "--"
        rows.append(" & ".join([tex(bname)] + cells + [delta]) + r" \\")
    macro = unc.get("macro_merged_minus_base")
    macro_s = f"{macro['diff_pp']:+.2f} pp (95\\% CI {macro['ci95'][0]:+.2f}, {macro['ci95'][1]:+.2f})" if macro else "--"
    fail = []
    for (m, b), r in sorted(summ.items()):
        fc = json.loads(r["failure_counts"])
        n = int(r["n_rows"])
        bad = sum(v for k, v in fc.items() if k in ("env_error", "no_result"))
        fail.append(f"{m}/{b} {100 * bad / max(1, n):.1f}\\%")

    body = rf"""\documentclass[9pt]{{extarticle}}
\usepackage[a4paper,margin=1.3cm]{{geometry}}
\usepackage{{xeCJK,booktabs,array}}
\setCJKmainfont{{Hiragino Sans GB}}
\setmainfont{{Helvetica Neue}}
\pagestyle{{empty}}\setlength{{\parskip}}{{3pt}}\setlength{{\parindent}}{{0pt}}
\begin{{document}}
{{\large\bfseries Qwen3.5-9B 多专家训练与模型合版实验}}\\[2pt]
{{\small github.com/ktwu01/qwen36-35b-takehome \quad 模型: huggingface.co/ktwu01}}

\textbf{{摘要}}\quad {text['abstract_zh']}

\textbf{{过程}}\quad {text['process_zh']}

\textbf{{结果}}\quad 主指标为百分制；TB2 为种子均值$\pm$标准差；最后一列为合版相对 Base 的配对 bootstrap 差值（百分点，95\% CI）。专家只评测其目标 benchmark（--）。

\begin{{center}}\small
\begin{{tabular}}{{l{'c' * len(MODELS)}c}}
\toprule
Benchmark & {' & '.join(n for _, n in MODELS)} & 合版$-$Base \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\textbf{{宏平均提升}}: {macro_s}。\quad \textbf{{运行失败率}}（环境失败与无结果，计入分母）: {', '.join(fail)}。

\textbf{{成本}}: {tex(cost.get('summary_zh', '见 logs/cost.json'))}

{text.get('notes_zh', '')}
\end{{document}}
"""
    Path("report").mkdir(exist_ok=True)
    Path("report/report.tex").write_text(body)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-output-directory=report", "report/report.tex"],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy("report/report.pdf", "report.pdf")
    pages = subprocess.run(["python3", "-c", "import re;d=open('report.pdf','rb').read();"
                            "print(len(re.findall(rb'/Type\\s*/Page[^s]', d)))"], capture_output=True, text=True).stdout
    print("report.pdf pages:", pages.strip())


if __name__ == "__main__":
    main()
