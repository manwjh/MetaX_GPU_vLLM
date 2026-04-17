#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/opt/muxi_gemma4_26B_TP1"
SCRIPT_DIR="${ROOT_DIR}/scripts"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-18010}}"
MODEL="${MODEL:-${SERVED_MODEL_NAME:-gemma-4-26B-A4B-it}}"
PROMPT="${PROMPT:-用一句话介绍长沙。}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TIMEOUT="${TIMEOUT:-300}"

python3 - <<PY
import json
import urllib.request

base = "${BASE_URL}".rstrip("/")
with urllib.request.urlopen(f"{base}/v1/models", timeout=15) as resp:
    print("health_HTTP", resp.status)
    payload = json.loads(resp.read().decode())
    print("health_models", [item.get("id") for item in payload.get("data", [])])
PY

exec python3 "${SCRIPT_DIR}/test_vllm_chat.py" \
  --url "${BASE_URL}/v1/chat/completions" \
  --api chat \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature 0 \
  --parse-gemma4 \
  --report-json \
  --timeout "${TIMEOUT}"
