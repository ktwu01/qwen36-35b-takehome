"""Write a mini-swe-agent text-based config for evaluation (SWE-Pro and ASI-Bench), identical for
every model: the given official config, unchanged except
  - the action fence name `mswea_bash_command` (mini-swe-agent v2) restored to `bash` (v1, and the
    format of the SWE expert's training trajectories), in every template and in model.action_regex;
  - model endpoint and sampling (temperature 1.0, top_p 0.95, top_k 20, seed 1).

    python scripts/make_swepro_config.py <source yaml> <served name> <api base> <out.yaml>
Sources: SWE-Pro uses SWE-bench_Pro-os@66f9276 v2/tooling/configs/mini_textbased.yaml; ASI-Bench uses
mini-swe-agent 2.4.6's builtin config/mini_textbased.yaml.
"""

import sys
from pathlib import Path

import yaml

src, name, api, out = sys.argv[1:5]
text = Path(src).read_text().replace("mswea_bash_command", "bash")
cfg = yaml.safe_load(text)
cfg.setdefault("model", {}).setdefault("model_kwargs", {}).update({
    "api_base": api, "api_key": "EMPTY", "temperature": 1.0, "top_p": 0.95, "seed": 1,
    "extra_body": {"top_k": 20},
})
cfg["model"]["action_regex"] = r"```bash\s*\n(.*?)\n```"
cfg["model"]["cost_tracking"] = "ignore_errors"  # local model has no litellm price entry
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(out)
