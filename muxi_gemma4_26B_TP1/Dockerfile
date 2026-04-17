ARG BASE_IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.15.0-maca.ai3.5.3.203-torch2.8-py310-ubuntu22.04-amd64
FROM ${BASE_IMAGE}

# Gemma4：权重 ``config.json`` 的 ``model_type=gemma4`` 需要 Transformers 已注册该架构。
# 部分沐曦基镜像内 Transformers 偏旧，会在 ``ModelConfig`` 阶段报错：
#   ``does not recognize this architecture ... gemma4``
# 长沙现场验证：``transformers==5.5.0`` + vLLM 0.15.0 可用。
ARG TRANSFORMERS_PIN=5.5.0
RUN /opt/conda/bin/pip install --no-cache-dir "transformers==${TRANSFORMERS_PIN}" \
  && /opt/conda/bin/python3 -c "import transformers; print('transformers_ok', transformers.__version__)"

WORKDIR /opt/muxi_gemma4_26B_TP1

COPY scripts/ /opt/muxi_gemma4_26B_TP1/scripts/
COPY patches/reasoning/ /opt/muxi_gemma4_26B_TP1/patches/reasoning/

RUN chmod +x /opt/muxi_gemma4_26B_TP1/scripts/container_entrypoint.sh \
    /opt/muxi_gemma4_26B_TP1/scripts/verify_service.sh \
    /opt/muxi_gemma4_26B_TP1/scripts/run_vllm_gemma4_tp2.sh

ENTRYPOINT ["/opt/muxi_gemma4_26B_TP1/scripts/container_entrypoint.sh"]
