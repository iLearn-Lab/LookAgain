#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

SCREENSPOT_DIR="${DATA_PATH:-evaluation/screenspot-pro}"
SCREENSPOT_ANNOTATIONS_DIR="${SCREENSPOT_DIR}/annotations"
SCREENSPOT_IMAGE_DIR="${SCREENSPOT_DIR}/images"
SCREENSPOT_SPLIT="overall"

CONVERTED_BASE="outputs/converted_data"
INPUT_JSON="${CONVERTED_BASE}/screenspot_pro_input.json"

mkdir -p "${CONVERTED_BASE}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python "${SUITE_ROOT}/data_transfer/run_multiturn_data_transfer.py" \
  --benchmark screenspot-pro \
  --annotations-dir "${SCREENSPOT_ANNOTATIONS_DIR}" \
  --image-dir "${SCREENSPOT_IMAGE_DIR}" \
  --split "${SCREENSPOT_SPLIT}" \
  --converted-json "${INPUT_JSON}" \
  --prepare-only

cat <<EOF

[prepared]
converted input:
  ${INPUT_JSON}
EOF
