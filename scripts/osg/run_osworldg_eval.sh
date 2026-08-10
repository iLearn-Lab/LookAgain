#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"
SUITE_ROOT="evaluation/grounding-agent-eval-suite"

API_BASE_URL="http://localhost:34567/v1"
API_MODEL_NAME="qwen3vl"

RUN_TAG="osworldg_full_and_refusal_$(date +%Y%m%d_%H%M%S)"
RESULT_BASE="outputs/log/${RUN_TAG}/run_results/"
FULL_OUTPUT_DIR="${RESULT_BASE}/osworld-g"
REFUSAL_OUTPUT_DIR="${RESULT_BASE}/osworld-g-refusal"
REFUSAL_SUMMARY_FILE="${REFUSAL_OUTPUT_DIR}/refusal_eval_summary.txt"

CONVERTED_BASE="outputs/converted_data"
FULL_INPUT_JSON="${CONVERTED_BASE}/osworld_g_sft_tool_call_input.json"
REFUSAL_INPUT_JSON="${CONVERTED_BASE}/osworld_g_refusal_sft_tool_call_input.json"

mkdir -p "${RESULT_BASE}" "${FULL_OUTPUT_DIR}" "${REFUSAL_OUTPUT_DIR}"

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

# ============= eval osg standard =============
echo "[start] osworld_g_full"
python evaluation/grounding-agent-eval-suite/grounding_agent/runner/run_sft_tool_call.py \
  "${FULL_INPUT_JSON}" \
  -o "${FULL_OUTPUT_DIR}" \
  "${COMMON_SFT_ARGS[@]}"
echo "[done] osworld_g_full"

# ============= eval osg refusal =============
echo "[start] osworldg_refusal"
python evaluation/grounding-agent-eval-suite/grounding_agent/runner/run_sft_tool_call.py \
  "${REFUSAL_INPUT_JSON}" \
  -o "${REFUSAL_OUTPUT_DIR}" \
  "${COMMON_SFT_ARGS[@]}"

python scripts/osg/score_osworldg_refusal.py \
  --results-json "${REFUSAL_OUTPUT_DIR}/results.json" \
  --summary-file "${REFUSAL_SUMMARY_FILE}"
echo "[done] osworldg_refusal"

cat <<EOF

[finished]
results:
  ${FULL_OUTPUT_DIR}
  ${REFUSAL_OUTPUT_DIR}

refusal summary:
  ${REFUSAL_SUMMARY_FILE}
EOF
