# `muxi_gemma4_26B_TP1`

wangjunhui@MacBook-Pro-5 changsha_gpu % python3 app/llm_tui/main.py --chat --stream
config source: /Users/wangjunhui/playcode/changsha_gpu/app/llm_tui/.env.example
chat mode started. input '/exit' to quit this mode.
stream: on
target: provider=openai_compatible base_url=http://***.***.82.8:18010 model=gemma-4-26B-A4B-it
health: http_status=200 latency_ms=40.37 models=1
max_tokens [128]: 
temperature [0.0]: 
you> 你是什么模型
assistant> 我是 Gemma 4，是由 Google DeepMind 开发的大型语言模型。我是一个开放权重（open weights）模型。
http_status: 200 latency_ms: 1951.82
usage: {}
you> 


本文档假定你当前位于**发布包根目录**：即解压 **`muxi_gemma4_26B_TP1.tar.gz`** 后得到的一级目录（目录名通常为 `muxi_gemma4_26B_TP1`）。下文简称「**本包**」。

**说明：** 本包是**独立交付物**——自洽的 Docker、脚本与补丁；**不包含** monorepo 里的 `workbench/` 研发区、实验笔记或 `dist/` 等构建机路径。若你是维护者、需要从源码仓库**重新打包**本 tarball，见仓库内 **`docs/muxi/RELEASING_muxi_gemma4_26B_TP1.md`**（该文件**不在**本 tarball 内，随完整仓库提供）。

这是一个面向 **MetaX / 沐曦 C500** 的 **Gemma4 26B IT 单卡 TP=1 清洁发布包**。目标不是保留研发过程，而是给出**最小可交付**的 Docker 方案，与已在长沙验证通过的基线对齐：

- `CUDA_VISIBLE_DEVICES=6`
- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=4096`
- `GPU_MEM_UTIL=0.88`
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
- `GEMMA4_DEFAULT_ENABLE_THINKING=0`

## 1. 包内内容

**目录地图（完整树状说明）见 `docs/LAYOUT.md`。**

| 路径 | 作用 |
|------|------|
| `Dockerfile` | 基于沐曦 `vllm-metax` 基镜像构建发布镜像 |
| `.env.example` | 建议环境变量模板 |
| `docs/LAYOUT.md` | 发布包目录结构与各目录职责 |
| `docs/QUICKSTART.md` | 给第一次试包的人用的 5 分钟上手说明 |
| `docs/OFFLINE_DISTRIBUTION.md` | `docker save/load` 离线分发说明 |
| `scripts/container_entrypoint.sh` | 容器启动入口：打补丁、安装 reasoning parser、启动 vLLM |
| `scripts/verify_service.sh` | 容器内验收：先查 `/v1/models`，再跑 chat 烟测 |
| `scripts/apply_vllm_site_patches.py` | 幂等 site-packages 补丁 |
| `scripts/run_vllm_gemma4_tp2.sh` | 实际启动底座，当前默认已是 TP=1；可选环境变量 **`VLLM_APISERVER_EXTRA`** 追加 `api_server` 参数（调度类调优） |
| `scripts/vllm_bootstrap_gemma4_maca.py` | Gemma4 on MACA bootstrap |
| `scripts/test_vllm_chat.py` | OpenAI chat/completions 烟测 |
| `patches/reasoning/*` | Gemma4 reasoning parser 发布版副本 |
| `patches/moe/vllm/.../moe.py` | 发布包内附带的 vLLM MoE 对照快照 |
| `patches/moe/patch_vllm_moe_skip_gemma4_fused.py` | 早期单点 MoE 补丁脚本，便于审计 |
| `patches/exported/changsha_gemma4_site_unified.diff` | 包含 `vllm moe.py` 与 `transformers/integrations/moe.py` 在内的统一 diff |
| `verification/` | 发版验证：见 `verification/reports/2026-04-17_release.md` 与 `evidence/` 下日志摘录 |
| `scripts/verify_release_package.sh` | **无 GPU** 静态检查：包内 `scripts/*.sh` 的 `bash -n` 与 `scripts/*.py` 的 `py_compile` |

## 2. 适用前提

需要满足：

1. 宿主机已经安装 Docker。
2. 宿主机可以访问沐曦基础镜像仓库，或者你已提前拿到同等基镜像。
3. 宿主机能挂载 Gemma4 模型目录到容器内，例如 `/data/gemma-4-26B-A4B-it`。
4. 宿主机上已有可用的 MACA / MetaX 运行环境，这个发布包**不是**从零安装驱动。
5. **必须用本包内 `Dockerfile` 构建镜像**（或等价地保证容器内 **Transformers 能识别 `model_type=gemma4`**）。发布镜像在构建阶段会 **`pip install transformers==5.5.0`**（可用 build-arg **`TRANSFORMERS_PIN`** 覆盖），避免基镜像自带 Transformers 过旧导致 `ModelConfig` 报错。

## 3. 构建镜像

建议先复制环境模板：

```bash
cp .env.example .env
```

然后构建：

```bash
source .env
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE_NAME}" \
  .
```

## 4. 启动服务

推荐命令：

```bash
source .env
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e PORT="${HOST_PORT}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}" \
  -e TRITON_DISABLE_SWIZZLE="${TRITON_DISABLE_SWIZZLE}" \
  -e REASONING_PARSER="${REASONING_PARSER}" \
  -e GEMMA4_DEFAULT_ENABLE_THINKING="${GEMMA4_DEFAULT_ENABLE_THINKING}" \
  -e GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS="${GEMMA4_USE_DEFAULT_CHAT_TEMPLATE_KWARGS}" \
  -e GEMMA4_USE_BOOTSTRAP="${GEMMA4_USE_BOOTSTRAP}" \
  -e VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
  -e VLLM_CHUNKED_PREFILL="${VLLM_CHUNKED_PREFILL}" \
  -v /your/model/dir:"${MODEL_PATH}" \
  "${IMAGE_NAME}"
```

查看日志：

```bash
docker logs -f "${CONTAINER_NAME}"
```

当日志中出现 `Application startup complete` 时，服务可用。

## 5. 测试方法

### 5.1 健康检查

宿主机执行：

```bash
curl -sS "http://127.0.0.1:${HOST_PORT}/v1/models"
```

最低通过标准：

- 返回 HTTP 200
- `data[0].id` 含 `gemma-4-26B-A4B-it`

### 5.2 容器内标准验收

```bash
docker exec "${CONTAINER_NAME}" /opt/muxi_gemma4_26B_TP1/scripts/verify_service.sh
```

最低通过标准：

- `health_HTTP 200`
- `http_status: 200`
- `message.content` 或客户端拆出的 `answer` 为正常中文

### 5.3 宿主机直接 chat 验收

```bash
curl -sS "http://127.0.0.1:${HOST_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26B-A4B-it","messages":[{"role":"user","content":"用一句话介绍长沙。"}],"max_tokens":64,"temperature":0}'
```

预期是能得到正常中文，而不是 `gy-gy-...` 之类碎片。

## 6. 推荐验收口径

建议分三层：

1. **服务层**：`/v1/models` 返回 200
2. **API 层**：`/v1/chat/completions` 返回 200
3. **质量层**：固定中文 prompt 下输出可读中文

如果你要和长沙现场当前里程碑对齐，推荐用下面这句作为最短验收标准：

> `muxi_gemma4_26B_TP1` 镜像启动后，单卡 `TP=1`、`MAX_MODEL_LEN=4096` 条件下，`/v1/chat/completions` 能稳定返回可读中文。

## 7. 当前建议参数与边界

当前推荐默认值：

- `TP=1`
- `MAX_MODEL_LEN=4096`
- `GPU_MEM_UTIL=0.88`

已知更激进但现场测过的值：

- `MAX_MODEL_LEN=8192`：可用
- `MAX_MODEL_LEN=16384`：可用
- `MAX_MODEL_LEN=32768`：当前组合下未在 ready timeout 内稳定起来
- `MAX_MODEL_LEN=65536`：当前组合下未在 ready timeout 内稳定起来

因此，**发布包默认值仍保持 4096**，把 `8192/16384` 视为后续运营可选项，而不是默认交付参数。

## 8. 验收记录目录 `verification/`

本目录用于**发版验证**：`reports/` 为结构化报告，`evidence/` 为可对照的输出与现场摘录。接收方**只需**阅读本包内文件；**不要求**持有完整研发仓库。

- **`verification/README.md`**：索引  
- **`verification/reports/2026-04-17_release.md`**：首轮发版验收摘要（含证据文件表）  
- **`verification/evidence/`**：`00_monorepo_make_check.log`、`01_package_static_verify.log`、`02_changsha_field_excerpt.txt`

## 9. 配套文档

- 目录总览：`docs/LAYOUT.md`
- 想最快试起来：`docs/QUICKSTART.md`
- 想给别人离线发包：`docs/OFFLINE_DISTRIBUTION.md`

## 10. 故障排查

### 10.1 `ModelConfig` / `gemma4` 架构不被识别

**现象（摘录）：**

```text
ValidationError: ... The checkpoint you are trying to load has model type `gemma4` but Transformers does not recognize this architecture.
```

**原因：** 容器内 **Transformers 版本过旧**，未注册 Gemma4。

**处理：**

1. 使用本目录 **`Dockerfile` 重新构建**（含 `pip install transformers==5.5.0`），**不要**只把 `tar.gz` 解压进任意旧镜像就当发布包用。
2. 自定义 **`BASE_IMAGE`** 时，构建仍可执行同一 `RUN pip install transformers==...`；若 `pip` 与 vLLM 依赖冲突，优先换用与长沙一致的 **`vllm-metax:0.15.0-...`** 基镜像再构建。
3. 日志里 **`trust_remote_code` is ignored** 为 vLLM 提示，可忽略，与上述错误无关。
4. **`docker build` 时 pip 提示 `vllm … requires transformers<5`，但安装了 `transformers 5.5.0`：** 来自 PyPI 元数据与 **Gemma4 需 Transformers 5.x** 的张力；沐曦 **`vllm-metax`** 镜像默认常为 **4.57.x**（无 `gemma4`），本 Dockerfile **有意**升到 **5.5.0**。长沙现场运行中的 **`wjh-vllm-gemma4`** 亦为 **5.5.0** 且与 vLLM 0.15 联调通过。
5. **发布镜像已在长沙宿主机实测（2026-04-17）：** `docker build -t muxi_gemma4_tp1_verify:local` 成功；对 **`muxi_gemma4_tp1_verify:local`** 执行 `AutoConfig.from_pretrained`（宿主机权重绑定 **`/8T/perfxcloud/model/google/gemma-4-26B-A4B-it` → 容器内 `/data/gemma-4-26B-A4B-it`**）输出 **`model_type=gemma4`**，证明 **不再出现** 用户反馈的 `ModelConfig` / 架构不可识别错误。完整起服与 API 烟测与线上一致，需挂载真实权重并占用 GPU，见 §4 与 `docs/QUICKSTART.md`。

### 10.2 端口与日志里的 `non-default args`

启动参数以日志为准；若你设置 **`PORT=18000`**，则监听 **18000**，与 `.env.example` 中的 **18010** 仅为示例，不是冲突。

---

## 11. 补丁审计说明

本包不仅带运行脚本，也带了**MoE 相关审计材料**，避免只看到 `apply_vllm_site_patches.py` 却看不到它改了什么：

- `patches/moe/vllm/model_executor/models/transformers/moe.py`
  当前发布基线使用的 **vLLM MoE 对照快照**（整文件随包附带）
- `patches/moe/patch_vllm_moe_skip_gemma4_fused.py`
  早期针对 Gemma4 `gate_up_proj` / `FusedMoE` 问题的单点 patch helper
- `patches/exported/changsha_gemma4_site_unified.diff`
  最完整的统一 diff，其中**明确包含**：
  - `vllm/model_executor/models/transformers/moe.py`
  - `transformers/integrations/moe.py`

说明：

- **`transformers/integrations/moe.py`** 在本包中以 **unified diff** 形式交付（未再单独附带完整快照文件）
- 运行路径仍以容器内 `scripts/apply_vllm_site_patches.py` 为准；上述文件用于**审计、复核、对照**

## 12. 边界说明

这个发布包：

- **包含**：Gemma4 TP=1 所需补丁、bootstrap、启动入口、基础验收脚本
- **不包含**：模型权重文件本身
- **不负责**：从零安装 MACA 驱动、Docker、本机内核与宿主机 GPU 驱动

如果后续你要把它做成真正对外发放的版本，我建议下一步再补：

1. 固定镜像 tag 的 `CHANGELOG`
2. 一份正式版本号约定，例如 `muxi_gemma4_26B_TP1:v2026.04.16`
