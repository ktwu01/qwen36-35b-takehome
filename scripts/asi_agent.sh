#!/bin/bash
# ASI-Bench file-exchange agent: mini-swe-agent (text-based bash actions) in the task workspace.
# Invoked by `asibench run --agent-cmd 'bash scripts/asi_agent.sh {workspace}'` with
#   ASI_MODEL (served name), ASI_API (OpenAI-compatible base URL), ASI_SEED, ASI_TRAJ_DIR.
# Same harness and text format as the SWE-Pro eval; identical settings for every model.
set -u
W=$1
BASE=${BASE:-/var/tmp/$USER}
MINI=$BASE/venvs/harbor/bin/mini
CFG=$("$BASE/venvs/harbor/bin/python" -c "import minisweagent,os;print(os.path.dirname(minisweagent.__file__))" 2>/dev/null | tail -1)/config/mini_textbased.yaml
ID=$(basename "$(dirname "$W")")
mkdir -p "$ASI_TRAJ_DIR"
cd "$W" || exit 1
export HOSTED_VLLM_API_KEY=EMPTY MSWEA_COST_TRACKING=ignore_errors
exec "$MINI" -m "hosted_vllm/$ASI_MODEL" -t "$(cat prompt.md)" -y -l 0 --exit-immediately \
  -c "$CFG" \
  -c "model.model_kwargs.api_base=$ASI_API" \
  -c model.model_kwargs.temperature=1.0 -c model.model_kwargs.top_p=0.95 \
  -c "model.model_kwargs.seed=$ASI_SEED" -c 'model.model_kwargs.extra_body={"top_k": 20}' \
  -c model.cost_tracking=ignore_errors \
  -c "environment.cwd=$W" -c environment.timeout=1800 \
  -o "$ASI_TRAJ_DIR/$ID.$(date +%s).traj.json" < /dev/null
