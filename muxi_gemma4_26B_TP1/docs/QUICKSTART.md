# `muxi_gemma4_26B_TP1` 5 分钟上手

这份文档给**第一次试包的人**用，只保留最短路径。

## 1. 你要先有的东西

1. 一台已经装好 Docker、MACA 运行环境的 MetaX 机器
2. Gemma4 权重目录，例如 `/data/gemma-4-26B-A4B-it`
3. 能拉取沐曦 `vllm-metax` 基镜像，或者本机已经有同等镜像

## 2. 构建镜像

```bash
cp .env.example .env
source .env
docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" .
```

**必须**执行上述 `docker build`：`Dockerfile` 内会 **`pip install transformers==5.5.0`**，使 Transformers 识别 `model_type=gemma4`。若只解压 `tar.gz` 把脚本拷进**旧容器**、跳过构建，会报 `ModelConfig` / `gemma4` 架构无法识别（见 `README.md` §10）。

## 3. 启动服务

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

推荐默认值就是：

- `CUDA_VISIBLE_DEVICES=6`
- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=4096`
- `GPU_MEM_UTIL=0.88`

## 4. 等服务起来

```bash
docker logs -f "${CONTAINER_NAME}"
```

看到：

```text
Application startup complete
```

说明服务 ready。

## 5. 最短验收

先查模型列表：

```bash
curl -sS "http://127.0.0.1:${HOST_PORT}/v1/models"
```

再发一条 chat：

```bash
curl -sS "http://127.0.0.1:${HOST_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26B-A4B-it","messages":[{"role":"user","content":"用一句话介绍长沙。"}],"max_tokens":64,"temperature":0}'
```

如果返回的是正常中文，就算通过。

## 6. 容器内标准验收

```bash
docker exec "${CONTAINER_NAME}" /opt/muxi_gemma4_26B_TP1/scripts/verify_service.sh
```

这一步会同时检查：

1. `/v1/models` 是否 200
2. `chat/completions` 是否 200
3. 输出是不是正常中文

## 7. 如果你只想“先跑起来”

不要一开始就改这些参数：

- 不要先改成 `TP=2`
- 不要先改 `FLEX_ATTENTION`
- 不要先把 `MAX_MODEL_LEN` 改到 `32768+`

先用默认值跑通，再做后续实验。
