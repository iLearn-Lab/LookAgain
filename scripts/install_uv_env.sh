#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"

export PATH="${HOME}/.local/bin:${PATH}"

pip install --user uv

uv venv "${VENV_DIR}" --python "3.12"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

uv pip install -U "vllm==0.19.1" --torch-backend="cu128"
uv pip install -U hf_transfer
uv pip install -r "evaluation/grounding-agent-eval-suite/requirements.txt"
