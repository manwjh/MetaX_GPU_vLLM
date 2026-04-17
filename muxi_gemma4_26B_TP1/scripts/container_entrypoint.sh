#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/opt/muxi_gemma4_26B_TP1"
SCRIPT_DIR="${ROOT_DIR}/scripts"
PATCH_DIR="${ROOT_DIR}/patches/reasoning"

if [[ -x /opt/conda/bin/python3 ]]; then
  PYTHON=/opt/conda/bin/python3
else
  PYTHON=python3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MODEL_PATH="${MODEL_PATH:-/data/gemma-4-26B-A4B-it}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-26B-A4B-it}"
export PORT="${PORT:-18010}"
export HOST="${HOST:-0.0.0.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export TRITON_DISABLE_SWIZZLE="${TRITON_DISABLE_SWIZZLE:-1}"
export REASONING_PARSER="${REASONING_PARSER:-gemma4}"
export GEMMA4_DEFAULT_ENABLE_THINKING="${GEMMA4_DEFAULT_ENABLE_THINKING:-0}"
export GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS="${GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS:-1}"
export GEMMA4_USE_BOOTSTRAP="${GEMMA4_USE_BOOTSTRAP:-1}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"
export VLLM_CHUNKED_PREFILL="${VLLM_CHUNKED_PREFILL:-0}"
export VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN="${VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN:-1}"
export VLLM_METAX_GEMMA3_27B_TILE32="${VLLM_METAX_GEMMA3_27B_TILE32:-0}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
if [[ "${VLLM_ENFORCE_EAGER}" != "1" ]]; then
  export MACA_GRAPH_LAUNCH_MODE="${MACA_GRAPH_LAUNCH_MODE:-1}"
fi

echo "[muxi_gemma4_26B_TP1] applying site patches"
"${PYTHON}" "${SCRIPT_DIR}/apply_vllm_site_patches.py"

echo "[muxi_gemma4_26B_TP1] installing reasoning parser"
"${PYTHON}" - <<'PY'
import os
import shutil
from pathlib import Path

import vllm

root = Path(os.path.dirname(vllm.__file__))
dst = root / "reasoning"
dst.mkdir(parents=True, exist_ok=True)
src = Path("/opt/muxi_gemma4_26B_TP1/patches/reasoning")
shutil.copyfile(src / "gemma4_reasoning_parser.py", dst / "gemma4_reasoning_parser.py")
shutil.copyfile(src / "__init__.py", dst / "__init__.py")
print("reasoning_installed_to", dst)
PY

echo "[muxi_gemma4_26B_TP1] starting vLLM"
exec "${SCRIPT_DIR}/run_vllm_gemma4_tp2.sh"
