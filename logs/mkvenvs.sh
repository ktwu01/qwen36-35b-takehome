set -x
export PIP_CACHE_DIR=$HOME/.pip-cache
/usr/bin/python3.12 -m venv ~/venvs/qwen-train && ~/venvs/qwen-train/bin/pip install -q -U pip && ~/venvs/qwen-train/bin/pip install -q -r logs/pg_venv_freeze.txt && ~/venvs/qwen-train/bin/pip install -q mergekit pandas pyarrow matplotlib scipy datasketch && echo TRAIN_OK
/usr/bin/python3.12 -m venv ~/venvs/harbor && ~/venvs/harbor/bin/pip install -q -U pip && ~/venvs/harbor/bin/pip install -q harbor==0.24.0 && echo HARBOR_OK
/usr/bin/python3.12 -m venv ~/venvs/asibench && ~/venvs/asibench/bin/pip install -q -U pip && ~/venvs/asibench/bin/pip install -q "asibench[full]==0.2.0" && echo ASI_OK
