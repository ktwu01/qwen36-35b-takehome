"""Build the one-page report.pdf (摘要 - 过程 - 结果) from the result files.

    python scripts/make_report.py       # needs xelatex + ctex; writes report/report.tex and report.pdf

Text: report/text.yaml (abstract_zh 150-200 chars, process_zh, notes_zh). Numbers: results/summary.csv,
analysis/uncertainty.json, logs/cost.json. Every number in the table comes from those files.
"""

import csv
import json
import shutil
import platform
import re
import subprocess
from pathlib import Path

import yaml

BENCH = [("swe_pro", "SWE-bench Pro v2-hard"), ("tb2", "Terminal-Bench 2.0"), ("hle", "HLE"), ("asi", "ASI-Bench")]
MODELS = [("base", "Base"), ("swe", "swe 专家"), ("term", "term 专家"), ("sci", "sci 专家"), ("merged", "合版")]
SCOPE = {"swe": "仅 SWE-Pro", "term": "仅 TB2", "sci": "仅 HLE/ASI"}  # experts: target benchmarks only


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def main():
    text = yaml.safe_load(Path("report/text.yaml").read_text())
    # a bare % in the prose would start a LaTeX comment and silently drop the rest of the line
    text = {k: re.sub(r"(?<!\\)%", r"\\%", v) if isinstance(v, str) else v for k, v in text.items()}
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
            if b == "tb2":
                n = unc["metrics"][f"{m}/tb2"]["n_complete_seeds"]
                v += f"$\\pm${float(r['seed_std']):.1f}" if n > 1 else ""
                v += f" ({n}种子)"
            cells.append(v)
        d = boot.get(b)
        delta = f"{d['diff_pp']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d else "--"
        rows.append(" & ".join([tex(bname)] + cells + [delta]) + r" \\")
    macro = unc.get("macro_merged_minus_base")
    macro_s = f"{macro['diff_pp']:+.2f} pp (95\\% CI {macro['ci95'][0]:+.2f}, {macro['ci95'][1]:+.2f})" if macro else "--"
    macro_abs = {m: sum(float(summ[(m, b)]["primary_percent"]) for b, _ in BENCH) / 4
                 for m in ("base", "merged") if all((m, b) in summ for b, _ in BENCH)}
    fail = []
    for (m, b), r in sorted(summ.items()):
        fc = json.loads(r["failure_counts"])
        n = int(r["n_rows"])
        bad = n - fc.get("ok", 0)  # every non-ok outcome (timeouts, env failures, no answer), as in README
        fail.append(f"{tex(m)}/{tex(b)} {100 * bad / max(1, n):.1f}\\%")
    seeds = []
    for m in ("base", "term", "merged"):
        t = unc["metrics"].get(f"{m}/tb2")
        if t:
            part = "，".join(f"{s[-1]} 号 {p['tasks_finished']}/89" for s, p in sorted(t["partial_seeds"].items()))
            seeds.append(f"{m} 完整 {t['n_complete_seeds']} 个" + (f"（未完成：{part}）" if part else ""))

    header = " & ".join(rf"\shortstack{{{n}\\\scriptsize {SCOPE[m]}}}" if m in SCOPE else n for m, n in MODELS)
    mac = platform.system() == "Darwin"
    hf = json.loads(Path("models/hf_index.json").read_text())  # published repos and revisions
    hf_links = r" \\ ".join(f"{k}: https://huggingface.co/{v['repo']} (revision {v['revision'][:7]})" for k, v in hf.items())
    CJK = "Hiragino Sans GB" if mac else "Noto Sans CJK SC"  # CJK font also covers Latin
    body = rf"""\documentclass[9pt]{{extarticle}}
\usepackage[a4paper,margin=1.3cm]{{geometry}}
\usepackage{{fontspec,booktabs,array}}
\setmainfont{{{CJK}}}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
\pagestyle{{empty}}\setlength{{\parskip}}{{9pt}}\setlength{{\parindent}}{{0pt}}
\begin{{document}}
{{\large\bfseries Qwen3.5-9B 多专家训练与模型合版实验}}\\[2pt]
{{\small 代码: https://github.com/ktwu01/qwen36-35b-takehome}}\\
{{\footnotesize {hf_links}}}

\textbf{{摘要}}\quad {text['abstract_zh']}

\textbf{{过程}}\quad {text['process_zh']}

\textbf{{结果}}\quad 主指标为百分制；TB2 为种子均值$\pm$标准差；最后一列为合版相对 Base 的配对 bootstrap 差值（百分点，95\% CI）。专家只评测其目标 benchmark（--）。

\begin{{center}}\small
\begin{{tabular}}{{l{'c' * len(MODELS)}c}}
\toprule
Benchmark & {header} & 合版$-$Base \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\textbf{{宏平均}}: Base {macro_abs.get('base', 0):.2f}，合版 {macro_abs.get('merged', 0):.2f}；提升 {macro_s}（目标 +3）。TB2 种子（题目要求 5 个）：{'；'.join(seeds)}，均值只用完整种子。\quad \textbf{{运行失败率}}（非正常结束：超时、环境失败、无答案，均计入分母）: {', '.join(fail)}。

\textbf{{成本}}: {tex(cost.get('summary_zh', '见 logs/cost.json'))}

{text.get('notes_zh', '')}
\end{{document}}
"""
    Path("report").mkdir(exist_ok=True)
    Path("report/report.tex").write_text(body)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-output-directory=report", "report/report.tex"],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy("report/report.pdf", "report.pdf")
    info = subprocess.run(["pdfinfo", "report.pdf"], capture_output=True, text=True).stdout
    print("report.pdf", next((l for l in info.splitlines() if l.startswith("Pages")), "Pages: ?"))


if __name__ == "__main__":
    main()
