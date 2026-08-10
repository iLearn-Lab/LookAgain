#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

MMBENCH_DIR="${DATA_PATH:-evaluation/mmbench-gui}"
MMBENCH_DATASET_JSON="${MMBENCH_DIR}/L2_annotations.json"
MMBENCH_IMAGE_DIR="${MMBENCH_DIR}/images"

CONVERTED_BASE="outputs/converted_data"
INPUT_JSON="${CONVERTED_BASE}/mmbench_gui_l2_input.json"

mkdir -p "${CONVERTED_BASE}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python "${SUITE_ROOT}/data_transfer/run_multiturn_data_transfer.py" \
  --benchmark mmbench-gui \
  --dataset-json "${MMBENCH_DATASET_JSON}" \
  --image-dir "${MMBENCH_IMAGE_DIR}" \
  --converted-json "${INPUT_JSON}" \
  --prepare-only

cat <<EOF

[prepared]
converted input:
  ${INPUT_JSON}
EOF
