#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/tiger/venv"

export CUDA_VISIBLE_DEVICES=0,1

TP_SIZE=2
PORT=34567
MODEL_PATH="${MODEL_PATH:-models/lookagain-8b}"
SERVED_MODEL_NAME="qwen3vl"
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=8
DTYPE="bfloat16"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

vllm serve "${MODEL_PATH}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --reasoning-parser qwen3 \
  --tensor-parallel-size "${TP_SIZE}" \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --tool-call-parser hermes \
  --tokenizer "${MODEL_PATH}" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 10}' \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice
