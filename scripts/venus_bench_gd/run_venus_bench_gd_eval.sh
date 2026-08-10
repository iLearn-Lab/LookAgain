#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

API_BASE_URL="http://localhost:34567/v1"
API_MODEL_NAME="qwen3vl"

RUN_TAG="venus_bench_gd_$(date +%Y%m%d_%H%M%S)"
RESULT_BASE="outputs/log/${RUN_TAG}/run_results"
OUTPUT_DIR="${RESULT_BASE}/venus-bench-gd"

CONVERTED_BASE="outputs/converted_data"
INPUT_JSON="${CONVERTED_BASE}/venus_bench_gd_input.json"

mkdir -p "${OUTPUT_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

COMMON_SFT_ARGS=(
  --workers 40
  --api-base "${API_BASE_URL}"
  --model "${API_MODEL_NAME}"
  --max-turns 5
  --max-tokens 1024
  --temperature 0.0
  --rollout-n 1
  --timeout 300
  --crop-size 512
  --tool-choice auto
  --run-name SFT_Model_ToolCall
  -v
)

echo "[start] venus_bench_gd"
python "${SUITE_ROOT}/grounding_agent/runner/run_sft_tool_call.py" \
  "${INPUT_JSON}" \
  -o "${OUTPUT_DIR}" \
  "${COMMON_SFT_ARGS[@]}"
echo "[done] venus_bench_gd"

cat <<EOF

[finished]
results:
  ${OUTPUT_DIR}
EOF
