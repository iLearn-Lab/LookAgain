#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

VENUS_DIR="${DATA_PATH:-evaluation/venus-bench-gd}"
VENUS_INSTRUCTION_DIR="${VENUS_DIR}/instruction"
VENUS_IMAGE_DIR="${VENUS_DIR}/images"
VENUS_META_JSON="${VENUS_DIR}/meta.json"
VENUS_INSTRUCTION_KEY="instruction"

CONVERTED_BASE="outputs/converted_data"
INPUT_JSON="${CONVERTED_BASE}/venus_bench_gd_input.json"

mkdir -p "${CONVERTED_BASE}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python "${SUITE_ROOT}/data_transfer/run_multiturn_data_transfer.py" \
  --benchmark venus-bench-gd \
  --instruction-dir "${VENUS_INSTRUCTION_DIR}" \
  --image-dir "${VENUS_IMAGE_DIR}" \
  --meta-json "${VENUS_META_JSON}" \
  --instruction-key "${VENUS_INSTRUCTION_KEY}" \
  --converted-json "${INPUT_JSON}" \
  --prepare-only

cat <<EOF

[prepared]
converted input:
  ${INPUT_JSON}
EOF
