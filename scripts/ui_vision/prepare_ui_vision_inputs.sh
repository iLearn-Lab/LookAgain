#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

UI_VISION_DIR="${DATA_PATH:-evaluation/ui-vision}"
UI_TASK_SETS="element"
UI_DATASET_JSON="${UI_VISION_DIR}/annotations/element_grounding/element_grounding_basic.json,${UI_VISION_DIR}/annotations/element_grounding/element_grounding_functional.json,${UI_VISION_DIR}/annotations/element_grounding/element_grounding_spatial.json"
UI_IMAGE_DIR="${UI_VISION_DIR}/images"

CONVERTED_BASE="outputs/converted_data"
INPUT_JSON="${CONVERTED_BASE}/ui_vision_input.json"

mkdir -p "${CONVERTED_BASE}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python "${SUITE_ROOT}/data_transfer/run_multiturn_data_transfer.py" \
  --benchmark ui-vision \
  --ui-task-sets "${UI_TASK_SETS}" \
  --dataset-json "${UI_DATASET_JSON}" \
  --image-dir "${UI_IMAGE_DIR}" \
  --converted-json "${INPUT_JSON}" \
  --prepare-only

cat <<EOF

[prepared]
converted input:
  ${INPUT_JSON}
EOF
