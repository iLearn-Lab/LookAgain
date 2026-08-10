#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

OSWORLD_DIR="${DATA_PATH:-evaluation/osworld-g}"
OSWORLD_DATASET_JSON="${OSWORLD_DIR}/OSWorld-G_refined.json"
OSWORLD_IMAGE_DIR="${OSWORLD_DIR}/images"

CONVERTED_BASE="outputs/converted_data"
FULL_INPUT_JSON="${CONVERTED_BASE}/osworld_g_sft_tool_call_input.json"
REFUSAL_FULL_JSON="${CONVERTED_BASE}/osworld_g_full_with_refusal_sft_tool_call_input.json"
REFUSAL_INPUT_JSON="${CONVERTED_BASE}/osworld_g_refusal_sft_tool_call_input.json"

mkdir -p "${CONVERTED_BASE}"

source "${VENV_DIR}/bin/activate"

python scripts/osg/prepare_osworldg_inputs.py \
  --suite-root "${SUITE_ROOT}" \
  --dataset-json "${OSWORLD_DATASET_JSON}" \
  --image-dir "${OSWORLD_IMAGE_DIR}" \
  --full-input-json "${FULL_INPUT_JSON}" \
  --refusal-full-json "${REFUSAL_FULL_JSON}" \
  --refusal-input-json "${REFUSAL_INPUT_JSON}"

cat <<EOF

[prepared]
converted inputs:
  ${FULL_INPUT_JSON}
  ${REFUSAL_FULL_JSON}
  ${REFUSAL_INPUT_JSON}
EOF
