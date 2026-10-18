"""Write the SWE-Pro agent config: the official v2/tooling/configs/mini_textbased.yaml
(SWE-bench_Pro-os@66f9276) unchanged except model endpoint and sampling, identical for every model.

    python scripts/make_swepro_config.py <sbp checkout> <served name> <api base> <out.yaml>
"""

import sys
from pathlib import Path

import yaml

sbp, name, api, out = sys.argv[1:5]
cfg = yaml.safe_load(Path(sbp, "v2/tooling/configs/mini_textbased.yaml").read_text())
cfg["model"]["model_kwargs"].update({
    "api_base": api, "api_key": "EMPTY", "temperature": 1.0, "top_p": 0.95, "seed": 1,
    "extra_body": {"top_k": 20},
})
cfg["model"]["cost_tracking"] = "ignore_errors"  # local model has no litellm price entry
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(out)
