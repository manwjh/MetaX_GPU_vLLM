#!/usr/bin/env bash
# Gemma4 26B + vLLM（沐曦 MetaX C500；当前默认 **TP=1**，可用 TENSOR_PARALLEL_SIZE=2/4 做对照）
#
# **显存预算（必读）：** 沐曦 C500 **单卡标称 64 GiB（65536 MiB）** VRAM，勿按公开资料里 **NVIDIA 80GB 档**
# 默认抄 ``--gpu-memory-utilization 0.92+``；本脚本默认 **0.88** 为框架/KV 校验留余量，OOM 或
# ``ValueError: Free memory on device`` 时再 **export GPU_MEM_UTIL=0.85** 等逐步降。
#
# 拓扑依据：宿主机 `mx-smi topo -m`（与 nvidia-smi topo 类似）
# - 8 卡分成两组「四卡域」：GPU0–3、GPU4–7；组与组之间多为 NODE（经 CPU/PCI 根复合体），
#   组内为 PXB/PIX。
# - 最近互联（PIX，单 PCIe 桥）：(0,1) 与 (6,7)。
# - 次优：同组内 PXB 对，例如 (2,3)、(4,5)、(0,2) 等。
# - 尽量避免：一张在 0–3、另一张在 4–7（链路为 NODE，AllReduce 延迟更高）。
# - 沐曦 C500 现场口径：卡间高速互联约 **100G**；**TP=4 时务必锁定同一四卡域内相邻四张**
#  （推荐 0,1,2,3 或 4,5,6,7），勿跨域跳号（如 0,1,4,5）。详见 Gemma4-四卡vLLM实验方案.md §0。
#
# 用法（宿主机先选卡，再带进容器）：
#   export CUDA_VISIBLE_DEVICES=6
#   docker exec -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" wjh-vllm-gemma4 \
#     bash -lc '/path/in/container/run_vllm_gemma4_tp2.sh'
#
# Gemma4 chat 模板含 thought 通道。**``host_one_shot_start_gemma4_tp2.sh``** 默认传入 **REASONING_PARSER=gemma4**
#（并同步 ``patches/stack_v015/vllm/reasoning/*``）。手工 ``docker exec`` 时自行 **export REASONING_PARSER=gemma4**；
# 若 EngineCore **KeyError**，先清空 **REASONING_PARSER** 再排查 reasoning 文件是否写入 site-packages。
#
# **``--default-chat-template-kwargs``**：默认 **``GEMMA4_DEFAULT_ENABLE_THINKING=0``** → ``enable_thinking:false``（沐曦上短输出更易在 ``content`` 出中文）；与 HF 官方「常开思考」演示不一致时属有意取舍。链式思考：**``export GEMMA4_DEFAULT_ENABLE_THINKING=1``**。完全不带该 CLI 参数：**``export GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS=0``**。请求级 **``chat_template_kwargs``** 仍覆盖服务端默认。
#
# `docker exec -d` 时建议写日志，否则无终端输出：
#   docker exec -d -e CUDA_VISIBLE_DEVICES=6 -e TRITON_DISABLE_SWIZZLE=1 \
#     -e VLLM_LOG_FILE=/tmp/vllm_gemma4.log -e PORT=18010 -e MAX_MODEL_LEN=4096 \
#     wjh-vllm-gemma4 bash /tmp/run_vllm_gemma4_tp2.sh
#
# 沐曦《AI 推理用户手册》：若改为图模式，需配合
# **export MACA_GRAPH_LAUNCH_MODE=1**（手册 MacaRT‑vLLM‑metax 章节）。
# 当前默认仍走 **--enforce-eager** 稳态基线；若要试性能优化，可设：
#   export VLLM_ENFORCE_EAGER=0
#   export MACA_GRAPH_LAUNCH_MODE=1
# 仅在已验证中文可读与稳定性不回退时，再考虑升级默认。
#
# 在长沙宿主机上不想拼命令时，用同目录 `host_one_shot_start_gemma4_tp2.sh`（一次执行）。

set -euo pipefail

_RUN_DIR="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VLLM_LOG_FILE:-}" ]]; then
  exec >>"${VLLM_LOG_FILE}" 2>&1
fi

# 同一容器内多次拉起会残留旧 bootstrap（环境变量与补丁世代不一致）；先收口再启动。
if command -v pkill >/dev/null 2>&1; then
  pkill -f vllm_bootstrap_gemma4_maca.py 2>/dev/null || true
  sleep 2
fi

# 容器内必须用 conda 的 python（`/usr/bin/python3` 常无 vllm）。
if [[ -x /opt/conda/bin/python3 ]]; then
  PYTHON=/opt/conda/bin/python3
else
  PYTHON=python3
fi

# 沐曦 + Gemma4：单条 user 的 chat 经 HF 模板渲染出的 prompt 与矩阵里 **无字面 bos** completions 前缀在 ``/tokenize`` 上不一致；
# vLLM ``renderers/hf.py`` 内可选替换为后者（**不改** 权重内 ``chat_template.jinja``）。关：**``export VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN=0``**。
: "${VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN:=1}"
export VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN

# MACA **triton_unified_attention**：Gemma4 的 SWA 层也是 (head_dim=128, sliding_window=1024)，与 Gemma3-27B 签名碰撞。
# 误把 Gemma4 当 Gemma3 走 tile=32 快路径会 **logits 错误**（见 ``GEMMA4_MACA_VLLM_METAX_v015.diff`` / ``_is_gemma3_attention``）。
# 仅当你明确在跑 **Gemma3-27B** 时才 ``export VLLM_METAX_GEMMA3_27B_TILE32=1``；**Gemma4 服务必须保持 0**。
: "${VLLM_METAX_GEMMA3_27B_TILE32:=0}"
export VLLM_METAX_GEMMA3_27B_TILE32

# Gemma4+MACA：在 import vLLM 之前打 transformers MoE / grouped_mm 补丁（见同目录 vllm_bootstrap_gemma4_maca.py）。
# EngineCore 子进程多为 spawn，仍以 **apply_vllm_site_patches.py 磁盘补丁** 为准；本入口保证 API 进程与 fork 路径一致。
# 沐曦镜像常预置 ``GEMMA4_USE_BOOTSTRAP=0``：``:="${GEMMA4_USE_BOOTSTRAP:=1}"`` **不会**覆盖已存在的 0，导致误走 ``python -m vllm``。Gemma4 默认 **强制** 走 bootstrap；确需纯 ``-m vllm`` 时在本脚本内改下一行或设 ``GEMMA4_ALLOW_NO_BOOTSTRAP=1`` 后再 ``export GEMMA4_USE_BOOTSTRAP=0``。
if [[ "${GEMMA4_ALLOW_NO_BOOTSTRAP:-0}" == "1" && "${GEMMA4_USE_BOOTSTRAP:-1}" == "0" ]]; then
  export GEMMA4_USE_BOOTSTRAP=0
else
  GEMMA4_USE_BOOTSTRAP=1
  export GEMMA4_USE_BOOTSTRAP
fi
BOOTSTRAP="${GEMMA4_BOOTSTRAP_SCRIPT:-${_RUN_DIR}/vllm_bootstrap_gemma4_maca.py}"

: "${MODEL_PATH:=/data/gemma-4-26B-A4B-it}"
# Client `model` field should match this (vLLM recipes use short HF-style id).
: "${SERVED_MODEL_NAME:=gemma-4-26B-A4B-it}"
: "${HOST:=0.0.0.0}"
: "${PORT:=18001}"
# Gemma 4 权重 config 常见 max_position_embeddings ≈ 262144（256k），--max-model-len 不得超过该值。
# KV 显存随上下文近似线性增长；TP=1 单卡基线默认 4096，TP=2/4 长窗更易 OOM。更长请 export 后压测，例如：
#   MAX_MODEL_LEN=32768  MAX_MODEL_LEN=65536  （失败则降或调 GPU_MEM_UTIL / KV 量化等沐曦支持项）
: "${MAX_MODEL_LEN:=4096}"
# C500 单卡 64GiB：0.96 易触发「单卡空闲显存不足」校验失败；0.90 仍偏紧，默认 0.88。
: "${GPU_MEM_UTIL:=0.88}"
: "${REASONING_PARSER:=}"
REASONING_ARGS=()
if [[ -n "${REASONING_PARSER}" ]]; then
  REASONING_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

: "${TENSOR_PARALLEL_SIZE:=1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "请设置 CUDA_VISIBLE_DEVICES，例如默认 TP=1: 6；TP=2 对照: 6,7；TP=4: 0,1,2,3（见脚本头部拓扑说明）" >&2
  exit 1
fi

_gpu_list="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a _gpus <<< "${_gpu_list}"
_n_gpus="${#_gpus[@]}"
if [[ "${_n_gpus}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES 解析失败: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "${TENSOR_PARALLEL_SIZE}" != "${_n_gpus}" ]]; then
  echo "CUDA_VISIBLE_DEVICES 中 GPU 数量（${_n_gpus}）须等于 TENSOR_PARALLEL_SIZE（${TENSOR_PARALLEL_SIZE}）" >&2
  exit 1
fi

# Gemma4+MACA：部分现场 chunked prefill 与异常续写相关；默认关闭。若需打开：export VLLM_CHUNKED_PREFILL=1
if [[ "${VLLM_CHUNKED_PREFILL:-0}" == "1" ]]; then
  _prefill_args=(--enable-chunked-prefill)
else
  _prefill_args=(--no-enable-chunked-prefill)
fi

# vLLM 0.15：`--attention-backend`（沐曦注册 Maca* 实现）。单变量对照时：`export VLLM_ATTENTION_BACKEND=FLASH_ATTN` 等。
# FLEX 仍 smem 顶穿时（日志 OutOfResources）：可先 **`export VLLM_METAX_FLEX_EXTRA_HALVE=1`**（见 vllm_metax `flex_attention.get_kernel_options` 与 **`GEMMA4_MACA_VLLM_METAX_v015.diff`**）。
_attn_args=()
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  _attn_args+=(--attention-backend "${VLLM_ATTENTION_BACKEND}")
fi

# Prefix KV 复用：在沐曦现场曾与「固定 prompt 下续写异常」并存；默认显式 **关闭**（与矩阵 ``cache_salt`` 互补）。
# 需要打开时：**``export VLLM_ENABLE_PREFIX_CACHING=1``**。
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  _prefix_cache_args=(--enable-prefix-caching)
else
  _prefix_cache_args=(--no-enable-prefix-caching)
fi

_VLLM_ARGS=(
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  "${_prefill_args[@]}"
  "${_prefix_cache_args[@]}"
  "${_attn_args[@]}"
  --limit-mm-per-prompt '{"image":0,"audio":0}'
  "${REASONING_ARGS[@]}"
)

: "${VLLM_ENFORCE_EAGER:=1}"
if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  _VLLM_ARGS+=(--enforce-eager)
else
  : "${MACA_GRAPH_LAUNCH_MODE:=1}"
  export MACA_GRAPH_LAUNCH_MODE
fi

# vLLM：``--default-chat-template-kwargs``。沐曦 + 固定中文可读性：默认 **关** thinking（``GEMMA4_DEFAULT_ENABLE_THINKING=0``），
# 减少短输出里 thought 占满、``content`` 侧难出中文；需要链式思考时再 **``export GEMMA4_DEFAULT_ENABLE_THINKING=1``**。
: "${GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS:=1}"
: "${GEMMA4_DEFAULT_ENABLE_THINKING:=0}"
if [[ "${GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS}" == "1" ]]; then
  if [[ "${GEMMA4_DEFAULT_ENABLE_THINKING}" == "1" ]]; then
    _VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":true}')
  else
    _VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":false}')
  fi
fi

# Optional: extra flags for ``vllm.entrypoints.openai.api_server`` (scheduler / batching / KV tuning).
# Example: ``export VLLM_APISERVER_EXTRA='--max-num-seqs 128 --max-num-batched-tokens 8192'``
# See ``Gemma4-TP1-吞吐深挖清单.md``.
if [[ -n "${VLLM_APISERVER_EXTRA:-}" ]]; then
  read -r -a _VLLM_APISERVER_EXTRA_ARR <<< "${VLLM_APISERVER_EXTRA}"
  _VLLM_ARGS+=("${_VLLM_APISERVER_EXTRA_ARR[@]}")
fi

if [[ "${GEMMA4_USE_BOOTSTRAP}" == "1" && -f "${BOOTSTRAP}" ]]; then
  exec "$PYTHON" "${BOOTSTRAP}" "${_VLLM_ARGS[@]}"
fi

if [[ "${GEMMA4_USE_BOOTSTRAP}" == "1" ]]; then
  echo "错误: GEMMA4_USE_BOOTSTRAP=1 但未找到 ${BOOTSTRAP}，拒绝回退为 python -m vllm（Gemma4 需 bootstrap；请 docker cp 同目录 vllm_bootstrap_gemma4_maca.py 至容器 /tmp）" >&2
  exit 1
fi

exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${_VLLM_ARGS[@]}"
