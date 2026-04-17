#!/usr/bin/env python3
"""
Idempotent on-disk patches for Gemma4 + vLLM 0.15 (Transformers backend) inside the container.

Run once per venv / after image rebuild:
  python3 apply_vllm_site_patches.py

Patches:
  1) vllm/.../transformers/moe.py  — top_k_experts for Gemma4 MoE config
  2) vllm/.../transformers/moe.py  — skip TransformersFusedMoE when gate_up_proj (packed MoE)
  3) vllm/.../models/utils.py      — load nn.Module buffers (layer_scalar)
  4) transformers/.../integrations/moe.py — _grouped_mm always uses grouped_mm_fallback
     (vLLM EngineCore spawn children ignore PYTHONSTARTUP monkeypatch)
  5) vllm/.../transformers/base.py — Gemma4 per-layer head_dim (sliding vs full) for Attention
  6) vllm_metax/.../triton_unified_attention.py — Gemma4 full head 512: TILE 16 + launch num_stages=1 (MACA 64KiB smem)
  7) vllm_metax/.../triton_unified_attention.py — disambiguate Gemma4 sliding (128,1024) vs Gemma3-27B tile-32 path (opt-in env)
  8) vllm/.../transformers/base.py — widen Gemma4 interleaved-head gate (``gemma4`` + ``global_head_dim``) on already-patched trees
  9) vllm_metax/.../flex_attention.py — second BLOCK_M/N halving when ``shared_memory_per_block_optin`` ≤ 64KiB (FLEX smem for head_dim 512)
 10) vllm_metax/.../flex_attention.py — optional third halving when ``VLLM_METAX_FLEX_EXTRA_HALVE=1`` (64KiB MACA 仍 OOR 时)
 11) vllm_metax/.../flex_attention.py — ``VLLM_METAX_FLEX_EAGER=1`` 跳过 ``torch.compile(flex_attention)``（避免 Inductor 生成 smem>65536 的 Triton kernel）
 12) vllm/.../transformers/base.py — Gemma4: ``k_proj``/``v_proj`` use ``colwise_rep`` under TP when flat KV shards are not head-aligned (fixes TP=4 with few KV heads)
 13) vllm/.../chat_completion/serving.py — call ``reasoning_parser.adjust_request`` before ``to_sampling_params`` (Gemma4: keep ``<|channel|>`` in decoded text for split)
  14) vllm/.../responses/serving.py — same for Responses API path
  15) vllm/.../chat_completion/serving.py — pass ``output_token_ids`` into ``extract_reasoning`` (Gemma4); ``TypeError`` fallback for other parsers
  16) vllm/.../responses/serving.py — same non-streaming extract path
  17) vllm/.../parser/responses_parser.py — same (when file exists)
  18) vllm/.../reasoning/gemma4_reasoning_parser.py — copy from this repo's ``vendor/vllm/...`` when present (token-id split + ``spaces_between_special_tokens``)
 19) vllm/.../renderers/hf.py — Gemma4+沐曦：单 user 或（白名单 system + user）的 chat 可选替换为与矩阵 completions 探针一致的手写前缀（``VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN``，**不改** HF ``chat_template.jinja``）
 20) vllm_metax/.../triton_unified_attention.py — Gemma4 full head：可选 ``VLLM_METAX_GEMMA4_FULL_HEAD_TILE=16|32`` 覆盖 TILE（``32`` 在部分 MACA 上可能 smem OOR）
 21) vllm_metax/.../triton_attn.py — Gemma4 full head：可选 ``VLLM_METAX_GEMMA4_FORCE_2D=1`` 强制走 fused 2D unified kernel（绕开 decode 分段 3D softmax 路径，用于 MACA A/B）
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent

# 与 vendor 中逻辑一致；仅在本脚本无法定位 ``vendor/vllm/.../hf.py`` 时使用。
_GEMMA4_METAX_CHAT_ALIGN_HELPER_FALLBACK = dedent(
    """
def _gemma4_metax_text_from_conversation_content(content: object) -> str | None:
    # metax_align_content_parts_v2
    _media_types = frozenset(
        {
            "image",
            "image_url",
            "input_image",
            "image_pil",
            "image_embeds",
            "audio_url",
            "input_audio",
            "audio_embeds",
            "video_url",
            "file",
        }
    )
    if isinstance(content, str):
        t = content.strip()
        return t if t else None
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                s = p.strip()
                if s:
                    parts.append(s)
                continue
            if not isinstance(p, dict):
                continue
            typ = p.get("type")
            if typ in _media_types:
                continue
            tx = p.get("text")
            if isinstance(tx, str) and tx.strip():
                parts.append(tx)
        joined = "".join(parts).strip()
        return joined if joined else None
    return None


def _gemma4_metax_system_message_matches(content: object, allowed: frozenset[str]) -> bool:
    tx = _gemma4_metax_text_from_conversation_content(content)
    return tx is not None and tx in allowed


def _gemma4_metax_chat_template_kwargs_for_safe_apply(
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    out = dict(kwargs)
    env = os.environ.get("VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN", "").strip().lower()
    if env in ("1", "true", "yes"):
        out["tokenize"] = False
    return out


def _maybe_gemma4_metax_align_chat_prompt_string(
    *,
    model_config: ModelConfig,
    prompt_raw: str | list[int],
    conversation: list[ConversationMessage],
) -> str | list[int]:
    if not isinstance(prompt_raw, str):
        return prompt_raw
    env = os.environ.get("VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN", "").strip().lower()
    if env not in ("1", "true", "yes"):
        return prompt_raw

    hf_cfg = getattr(model_config, "hf_config", None)
    if hf_cfg is None:
        return prompt_raw
    text_cfg = getattr(model_config, "hf_text_config", None)
    model_type = getattr(hf_cfg, "model_type", None) or (
        getattr(text_cfg, "model_type", None) if text_cfg is not None else None
    )
    archs: list[str] = list(getattr(hf_cfg, "architectures", None) or [])
    if text_cfg is not None:
        archs.extend(getattr(text_cfg, "architectures", None) or [])
    is_gemma4 = model_type in ("gemma4", "gemma4_text") or any(
        "Gemma4" in str(a) for a in archs
    )
    if not is_gemma4:
        mid = str(getattr(model_config, "model", "") or "").lower()
        if "gemma-4" not in mid and "gemma4" not in mid:
            return prompt_raw

    _METAX_ALIGNABLE_SYSTEM_MESSAGES = frozenset({"You are a helpful assistant."})

    u: str | None = None
    if len(conversation) == 1:
        m0 = conversation[0]
        if m0.get("role") != "user":
            return prompt_raw
        u = _gemma4_metax_text_from_conversation_content(m0.get("content"))
    elif len(conversation) == 2:
        m0, m1 = conversation[0], conversation[1]
        if m0.get("role") != "system" or m1.get("role") != "user":
            return prompt_raw
        if not _gemma4_metax_system_message_matches(
            m0.get("content"), _METAX_ALIGNABLE_SYSTEM_MESSAGES
        ):
            return prompt_raw
        u = _gemma4_metax_text_from_conversation_content(m1.get("content"))
    else:
        return prompt_raw

    if u is None:
        return prompt_raw

    nl = chr(10)
    aligned = "<|turn>user" + nl + u + " " + nl + "<|turn>model" + nl + "<|channel>thought" + nl + " "
    logger.info(
        "Gemma4 Metax chat prompt align: len %d -> %d (messages=%d)",
        len(prompt_raw),
        len(aligned),
        len(conversation),
    )
    return aligned


"""
).strip() + "\n\n"


def _gemma4_metax_hf_helper_src() -> str:
    """容器 ``/tmp/apply_…`` 时读不到 vendor：先试仓库相对路径，否则用 fallback。"""
    here = Path(__file__).resolve().parent
    for vendor_hf in (
        here.parent / "vendor/vllm/vllm/renderers/hf.py",
        here.parent.parent / "vendor/vllm/vllm/renderers/hf.py",
    ):
        if not vendor_hf.is_file():
            continue
        vt = vendor_hf.read_text()
        s_txt = vt.find("def _gemma4_metax_text_from_conversation_content")
        s_kw = vt.find("def _gemma4_metax_chat_template_kwargs_for_safe_apply")
        s_m = vt.find("def _maybe_gemma4_metax_align_chat_prompt_string")
        if s_m < 0:
            continue
        if s_txt >= 0:
            s = s_txt
        elif s_kw >= 0:
            s = s_kw
        else:
            s = s_m
        e = vt.find("\n\nclass HfRenderer", s_m)
        if s >= 0 and e > s:
            return vt[s:e].strip() + "\n\n"
    return _GEMMA4_METAX_CHAT_ALIGN_HELPER_FALLBACK


def _site() -> Path:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
        return root
    except Exception as e:
        print("Import vllm first (same env as server):", e, file=sys.stderr)
        sys.exit(1)


def patch_moe_topk(moe_py: Path) -> str:
    text = moe_py.read_text()
    if '["num_experts_per_tok", "top_k", "top_k_experts"]' in text:
        return "moe top_k: already patched"
    old = '["num_experts_per_tok", "top_k"], None)'
    new = '["num_experts_per_tok", "top_k", "top_k_experts"], None)'
    if old not in text:
        return "moe top_k: anchor missing (unexpected vLLM version)"
    bak = moe_py.with_suffix(".py.bak.gemma4_topk")
    if not bak.exists():
        bak.write_text(text)
    moe_py.write_text(text.replace(old, new, 1))
    return "moe top_k: applied"


def patch_moe_skip_fused(moe_py: Path) -> str:
    text = moe_py.read_text()
    marker = "not load these checkpoints"
    if marker in text:
        return "moe skip_fused: already patched"
    old = """                if child_name == "experts" and (is_modulelist or is_3d):
                    # Alias for readability
                    mlp = module
                    experts = child_module"""
    new = """                if child_name == "experts" and (is_modulelist or is_3d):
                    # Gemma4 uses fused gate_up_proj + down_proj tensors; FusedMoE's
                    # expert_mapping expects Mixtral-style per-shard names and will
                    # not load these checkpoints. Keep native Transformers experts.
                    if any(
                        "gate_up_proj" in n
                        for n, _ in child_module.named_parameters()
                    ):
                        _recursive_replace(child_module, prefix=qual_name)
                        continue
                    # Alias for readability
                    mlp = module
                    experts = child_module"""
    if old not in text:
        return "moe skip_fused: anchor missing (already different layout?)"
    bak = moe_py.with_suffix(".py.bak.skip_gemma4_fused")
    if not bak.exists():
        bak.write_text(text)
    moe_py.write_text(text.replace(old, new, 1))
    return "moe skip_fused: applied"


def maybe_restore_moe_skip_fused(moe_py: Path) -> str:
    """
    Optional experiment mode:
      GEMMA4_ALLOW_FUSED_MOE=1
    Revert skip_fused patch from backup so Gemma4 can attempt FusedMoE path.
    """
    allow = os.environ.get("GEMMA4_ALLOW_FUSED_MOE", "0").strip().lower()
    if allow not in ("1", "true", "yes"):
        return "moe skip_fused: keep patched (default)"
    bak = moe_py.with_suffix(".py.bak.skip_gemma4_fused")
    if not bak.is_file():
        return "moe skip_fused: allow_fused requested but backup missing (keep current)"
    try:
        moe_py.write_text(bak.read_text())
    except Exception as e:
        return f"moe skip_fused: allow_fused restore failed ({e})"
    return "moe skip_fused: restored backup (allow_fused=1)"


def patch_utils_buffers(utils_py: Path) -> str:
    text = utils_py.read_text()
    if "child_buffers = dict(module.named_buffers(recurse=False))" in text:
        return "utils buffers: already patched"
    old = """        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:"""
    new = """        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))
        child_buffers = dict(module.named_buffers(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:"""
    if old not in text:
        return "utils buffers: anchor1 missing"
    old2 = """            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:"""
    new2 = """            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            elif child_prefix in child_buffers:
                buffer = child_buffers[child_prefix]
                if self._can_skip(prefix):
                    logger.debug("Skipping buffer %s", prefix)
                    continue
                for weight_name, weight_data in child_weights:
                    weight_qualname = self._get_qualname(base_prefix, weight_name)
                    if self._can_skip(weight_qualname):
                        logger.debug("Skipping weight %s", weight_qualname)
                        continue
                    if weight_name != "":
                        if self._can_ignore_unexpected(weight_qualname):
                            logger.debug("Ignoring weight %s", weight_qualname)
                            continue
                        raise ValueError(
                            f"Attempted to load nested weight {weight_qualname!r} "
                            f"into a single buffer {base_prefix!r}"
                        )
                    default_weight_loader(buffer, weight_data)
                    logger.debug(
                        "Loaded buffer %s with shape %s",
                        weight_qualname,
                        tuple(buffer.shape),
                    )
                    yield weight_qualname
            else:"""
    if old2 not in text:
        return "utils buffers: anchor2 missing"
    bak = utils_py.with_suffix(".py.bak.layer_scalar")
    if not bak.exists():
        bak.write_text(text)
    text = text.replace(old, new, 1).replace(old2, new2, 1)
    utils_py.write_text(text)
    return "utils buffers: applied"


def patch_transformers_moe_maca(tf_moe_py: Path) -> str:
    text = tf_moe_py.read_text()
    if "MUXI/MACA: torch._grouped_mm" in text:
        return "transformers integrations/moe: already patched"
    old = """    if _can_use_grouped_mm(input, weight, offs):
        # torch.nn.functional.grouped_mm and torch._grouped_mm are not autocast-enabled,
        # when autocast is enabled we can end up with intermediate tensors in fp32 (e.g. LayerNorm output) and weight tensors in bf16
        # In that case we need to cast the input to the weight dtype to avoid dtype mismatch errors.
        # See: https://github.com/pytorch/pytorch/issues/174763
        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.nn.functional.grouped_mm(input.to(weight.dtype), weight, offs=offs)
        elif hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(input.to(weight.dtype), weight, offs=offs)

    return torch.ops.transformers.grouped_mm_fallback(input, weight, offs=offs)"""
    new = """    # MUXI/MACA: torch._grouped_mm / functional.grouped_mm require NV sm90 on this PyTorch build.
    return torch.ops.transformers.grouped_mm_fallback(input.to(weight.dtype), weight, offs=offs)"""
    if old not in text:
        return "transformers integrations/moe: anchor missing (unexpected transformers version)"
    bak = tf_moe_py.with_suffix(".py.bak.maca_grouped_mm")
    if not bak.exists():
        bak.write_text(text)
    tf_moe_py.write_text(text.replace(old, new, 1))
    return "transformers integrations/moe: applied"


def patch_vllm_base_gemma4_attention(base_py: Path) -> str:
    text = base_py.read_text()
    if "_gemma4_alt" in text:
        return "vllm base create_attention_instances: already patched"
    old = """        attention_instances = {}
        for i in range(start, end):
            # Handle interleaved sliding window attention
            per_layer_sliding_window = None
            if (
                hasattr(self.config, "layer_types")
                and self.config.layer_types[i] == "sliding_attention"
            ):
                per_layer_sliding_window = self.config.sliding_window

            attn_cls = (
                EncoderOnlyAttention
                if attn_type == AttentionType.ENCODER_ONLY
                else Attention
            )
            attention_instances[i] = attn_cls(
                num_heads=num_heads,
                head_size=head_size,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=head_size**-0.5,
                num_kv_heads=num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                logits_soft_cap=logits_soft_cap,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{i}.attn",
                attn_type=attn_type,
            )
        return attention_instances"""
    new = """        attention_instances = {}
        for i in range(start, end):
            layer_head_size = head_size
            layer_num_kv_heads = num_kv_heads
            per_layer_sliding_window = None

            tc_layer_types = getattr(text_config, "layer_types", None)
            _text_mt = getattr(text_config, "model_type", None)
            _gemma4_alt = _text_mt in ("gemma4_text", "gemma4") and getattr(
                text_config, "global_head_dim", None
            ) is not None
            if (
                _gemma4_alt
                and tc_layer_types is not None
                and i < len(tc_layer_types)
            ):
                is_sliding_layer = tc_layer_types[i] == "sliding_attention"
                layer_head_size = (
                    text_config.head_dim
                    if is_sliding_layer
                    else text_config.global_head_dim
                )
                use_alt_kv = text_config.attention_k_eq_v and not is_sliding_layer
                n_kv = text_config.num_key_value_heads
                if use_alt_kv and getattr(
                    text_config, "num_global_key_value_heads", None
                ) is not None:
                    n_kv = text_config.num_global_key_value_heads
                tp = self.parallel_config.tensor_parallel_size
                layer_num_kv_heads = max(1, n_kv // tp)
                if is_sliding_layer:
                    per_layer_sliding_window = text_config.sliding_window
            else:
                if (
                    hasattr(self.config, "layer_types")
                    and i < len(self.config.layer_types)
                    and self.config.layer_types[i] == "sliding_attention"
                ):
                    per_layer_sliding_window = self.config.sliding_window

            attn_cls = (
                EncoderOnlyAttention
                if attn_type == AttentionType.ENCODER_ONLY
                else Attention
            )
            attention_instances[i] = attn_cls(
                num_heads=num_heads,
                head_size=layer_head_size,
                scale=layer_head_size**-0.5,
                num_kv_heads=layer_num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                logits_soft_cap=logits_soft_cap,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{i}.attn",
                attn_type=attn_type,
            )
        return attention_instances"""
    if old not in text:
        return "vllm base: anchor missing (unexpected vLLM version)"
    bak = base_py.with_suffix(".py.bak.gemma4_attn")
    if not bak.exists():
        bak.write_text(text)
    base_py.write_text(text.replace(old, new, 1))
    return "vllm base create_attention_instances: applied"


def patch_vllm_base_gemma4_kv_gather_colwise(base_py: Path) -> str:
    """Gemma4 GQA: few KV heads + column-parallel without gather breaks HF .view(..., head_dim) at high TP."""
    text = base_py.read_text()
    if "Gemma4: global attention uses few KV heads" in text:
        return "vllm base gemma4_kv_gather: already patched"
    old = """                    style = tp_plan.get(pattern, "replicate")
                    new_module = replace_linear_class(
                        child_module, style, self.quant_config, prefix=qual_name
                    )"""
    new = """                    style = tp_plan.get(pattern, "replicate")
                    # Gemma4: global attention uses few KV heads (e.g. 2). Column-parallel
                    # splits the flat out_features dimension; when num_kv_heads is not a
                    # multiple of tensor_parallel_size, each rank's shard is not an integer
                    # number of head_dim-wide heads, so HF's .view(..., -1, head_dim) on
                    # k_proj/v_proj output fails. Use gather_output for those projections.
                    _text_mt = getattr(self.text_config, "model_type", None)
                    if (
                        style == "colwise"
                        and self.tp_group.world_size > 1
                        and _text_mt in ("gemma4", "gemma4_text")
                        and re.search(
                            r"layers\\.\\d+\\.self_attn\\.(k_proj|v_proj)$", qual_name
                        )
                        is not None
                    ):
                        style = "colwise_rep"
                    new_module = replace_linear_class(
                        child_module, style, self.quant_config, prefix=qual_name
                    )"""
    if old not in text:
        return "vllm base gemma4_kv_gather: anchor missing (unexpected vLLM layout)"
    bak = base_py.with_suffix(".py.bak.gemma4_kv_gather")
    if not bak.exists():
        bak.write_text(text)
    base_py.write_text(text.replace(old, new, 1))
    return "vllm base gemma4_kv_gather: applied"


def patch_vllm_chat_completion_reasoning_adjust(chat_serving_py: Path) -> str:
    """Wire Gemma4ReasoningParser.adjust_request so SamplingParams skip_special_tokens=False."""
    text = chat_serving_py.read_text()
    if "Reasoning parsers (e.g. Gemma4) may define" in text:
        return "vllm chat_completion reasoning adjust: already patched"
    old = """        conversation, engine_prompts = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )"""
    new = """        conversation, engine_prompts = result

        # Reasoning parsers (e.g. Gemma4) may define ``adjust_request`` to set
        # ``skip_special_tokens=False`` so channel delimiter text survives
        # detokenization; ``SamplingParams`` must see that before ``to_sampling_params``.
        if self.reasoning_parser and self.renderer.tokenizer is not None:
            try:
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser(
                    self.renderer.tokenizer,
                    chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                )
                adjust_req = getattr(reasoning_parser, "adjust_request", None)
                if adjust_req is not None:
                    request = adjust_req(request)
            except RuntimeError:
                logger.exception(
                    "Reasoning parser adjust_request skipped (initialization failed)"
                )

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )"""
    if old not in text:
        return "vllm chat_completion reasoning adjust: anchor missing"
    bak = chat_serving_py.with_suffix(".py.bak.reasoning_adjust")
    if not bak.exists():
        bak.write_text(text)
    chat_serving_py.write_text(text.replace(old, new, 1))
    return "vllm chat_completion reasoning adjust: applied"


def patch_vllm_responses_reasoning_adjust(resp_serving_py: Path) -> str:
    """Responses API: same adjust_request before to_sampling_params."""
    text = resp_serving_py.read_text()
    if "req_ctkw = getattr(request, \"chat_template_kwargs\"" in text:
        return "vllm responses reasoning adjust: already patched"
    old = """            else:
                messages, engine_prompts = await self._make_request(
                    request, prev_response, renderer
                )

        except (
            ValueError,
            TypeError,
            RuntimeError,
            jinja2.TemplateError,
            NotImplementedError,
        ) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)"""
    new = """            else:
                messages, engine_prompts = await self._make_request(
                    request, prev_response, renderer
                )

            if self.reasoning_parser and tokenizer is not None:
                try:
                    req_ctkw = getattr(request, "chat_template_kwargs", None)
                    chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                        req_ctkw,
                        self.default_chat_template_kwargs,
                    )
                    reasoning_parser = self.reasoning_parser(
                        tokenizer,
                        chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                    )
                    adjust = getattr(reasoning_parser, "adjust_request", None)
                    if adjust is not None:
                        request = adjust(request)
                except RuntimeError:
                    logger.exception(
                        "Reasoning parser adjust_request skipped (initialization failed)"
                    )

        except (
            ValueError,
            TypeError,
            RuntimeError,
            jinja2.TemplateError,
            NotImplementedError,
        ) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)"""
    if old not in text:
        return "vllm responses reasoning adjust: anchor missing"
    bak = resp_serving_py.with_suffix(".py.bak.reasoning_adjust_resp")
    if not bak.exists():
        bak.write_text(text)
    resp_serving_py.write_text(text.replace(old, new, 1))
    return "vllm responses reasoning adjust: applied"


def patch_vllm_chat_completion_reasoning_extract_ids(chat_serving_py: Path) -> str:
    """Non-streaming chat: pass completion ``token_ids`` into ``extract_reasoning`` when supported."""
    text = chat_serving_py.read_text()
    if "oids = output.token_ids" in text:
        return "vllm chat_completion reasoning extract_ids: already patched"
    old = """                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )"""
    new = """                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                oids = output.token_ids
                try:
                    reasoning, content = reasoning_parser.extract_reasoning(
                        output.text,
                        request=request,
                        output_token_ids=list(oids) if oids is not None else None,
                    )
                except TypeError:
                    reasoning, content = reasoning_parser.extract_reasoning(
                        output.text, request=request
                    )"""
    if old not in text:
        return "vllm chat_completion reasoning extract_ids: anchor missing"
    bak = chat_serving_py.with_suffix(".py.bak.reasoning_extract_ids")
    if not bak.exists():
        bak.write_text(text)
    chat_serving_py.write_text(text.replace(old, new, 1))
    return "vllm chat_completion reasoning extract_ids: applied"


def patch_vllm_responses_reasoning_extract_ids(resp_serving_py: Path) -> str:
    """Responses ``_make_response_output_items``: same ``output_token_ids`` extract path."""
    text = resp_serving_py.read_text()
    if "oids = final_output.token_ids" in text:
        return "vllm responses reasoning extract_ids: already patched"
    old = """            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                raise e

            reasoning, content = reasoning_parser.extract_reasoning(
                final_output.text, request=request
            )"""
    new = """            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                raise e

            oids = final_output.token_ids
            try:
                reasoning, content = reasoning_parser.extract_reasoning(
                    final_output.text,
                    request=request,
                    output_token_ids=list(oids) if oids is not None else None,
                )
            except TypeError:
                reasoning, content = reasoning_parser.extract_reasoning(
                    final_output.text, request=request
                )"""
    if old not in text:
        return "vllm responses reasoning extract_ids: anchor missing"
    bak = resp_serving_py.with_suffix(".py.bak.reasoning_extract_ids_resp")
    if not bak.exists():
        bak.write_text(text)
    resp_serving_py.write_text(text.replace(old, new, 1))
    return "vllm responses reasoning extract_ids: applied"


def patch_vllm_responses_parser_reasoning_extract_ids(parser_py: Path) -> str:
    """``ResponsesParser.process``: same optional ``output_token_ids``."""
    text = parser_py.read_text()
    if "oids = output.token_ids" in text and "reasoning_parser_instance" in text:
        return "vllm responses_parser reasoning extract_ids: already patched"
    old = """        reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
            output.text, request=self.request
        )"""
    new = """        oids = output.token_ids
        try:
            reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
                output.text,
                request=self.request,
                output_token_ids=list(oids) if oids is not None else None,
            )
        except TypeError:
            reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
                output.text, request=self.request
            )"""
    if old not in text:
        return "vllm responses_parser reasoning extract_ids: anchor missing"
    bak = parser_py.with_suffix(".py.bak.reasoning_extract_ids_rparser")
    if not bak.exists():
        bak.write_text(text)
    parser_py.write_text(text.replace(old, new, 1))
    return "vllm responses_parser reasoning extract_ids: applied"


def patch_vllm_hf_gemma4_metax_chat_prompt_align(hf_py: Path) -> str:
    """Inject / 升级 Gemma4 Metax chat 对齐（含 ``tokenize=False`` 强制走字符串替换）。"""
    text = hf_py.read_text()
    helper = _gemma4_metax_hf_helper_src()
    _mid_gate = 'gemma-4" not in mid'
    _openai_content = "def _gemma4_metax_text_from_conversation_content"
    _content_v2 = "metax_align_content_parts_v2"
    kw_fn = "def _gemma4_metax_chat_template_kwargs_for_safe_apply"
    old_kw = """            conversation,
            **kwargs,
        )
        prompt_raw = _maybe_gemma4_metax_align_chat_prompt_string("""
    new_kw = """            conversation,
            **_gemma4_metax_chat_template_kwargs_for_safe_apply(kwargs),
        )
        prompt_raw = _maybe_gemma4_metax_align_chat_prompt_string("""
    notes: list[str] = []

    if "_maybe_gemma4_metax_align_chat_prompt_string" in text:
        dirty = False
        if _mid_gate not in text or _openai_content not in text or _content_v2 not in text:
            s_m = text.find("def _maybe_gemma4_metax_align_chat_prompt_string")
            if s_m < 0:
                return "vllm hf gemma4 metax chat align: upgrade anchor missing"
            pos_ok = [
                p
                for p in (
                    text.find(_openai_content),
                    text.find(kw_fn),
                    s_m,
                )
                if p >= 0
            ]
            s = min(pos_ok)
            e = text.find("\n\nclass HfRenderer", s_m)
            if e < 0:
                return "vllm hf gemma4 metax chat align: upgrade anchor missing"
            bak = hf_py.with_suffix(hf_py.suffix + ".bak.gemma4_metax_chat_align_upgrade")
            if hf_py.is_file() and not bak.exists():
                shutil.copy2(hf_py, bak)
            text = text[:s] + helper + text[e:]
            dirty = True
            notes.append("helper upgraded")
        if kw_fn not in text:
            kw_helper = (
                "def _gemma4_metax_chat_template_kwargs_for_safe_apply(\n"
                "    kwargs: dict[str, Any],\n"
                ") -> dict[str, Any]:\n"
                "    out = dict(kwargs)\n"
                "    env = os.environ.get(\"VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN\", \"\").strip().lower()\n"
                "    if env in (\"1\", \"true\", \"yes\"):\n"
                "        out[\"tokenize\"] = False\n"
                "    return out\n\n"
            )
            s_ins = text.find("def _maybe_gemma4_metax_align_chat_prompt_string")
            if s_ins < 0:
                return "vllm hf gemma4 metax chat align: insert kwargs helper anchor missing"
            text = text[:s_ins] + kw_helper + text[s_ins:]
            dirty = True
            notes.append("kwargs helper inserted")
        if old_kw in text:
            nkw = text.count(old_kw)
            if nkw != 2:
                return f"vllm hf gemma4 metax chat align: kwargs wire sites {nkw} != 2"
            text = text.replace(old_kw, new_kw, 2)
            dirty = True
            notes.append("safe_apply wired")
        if dirty:
            bak_kw = hf_py.with_suffix(hf_py.suffix + ".bak.gemma4_metax_chat_align_kw")
            if hf_py.is_file() and not bak_kw.exists():
                shutil.copy2(hf_py, bak_kw)
            hf_py.write_text(text)
            return "vllm hf gemma4 metax chat align: " + ", ".join(notes)
        if (
            _mid_gate in text
            and _openai_content in text
            and _content_v2 in text
            and kw_fn in text
            and old_kw not in text
        ):
            return "vllm hf gemma4 metax chat align: already patched"
        return "vllm hf gemma4 metax chat align: incomplete (unexpected hf.py state)"

    anchor = "\n\nclass HfRenderer(RendererLike):"
    if anchor not in text:
        return "vllm hf gemma4 metax chat align: class anchor missing"

    head = "\n".join(text.splitlines()[:20])
    if "import os" not in head and "import inspect" in text:
        text = text.replace("import inspect\n", "import inspect\nimport os\n", 1)

    inject = "\n\n" + helper + "class HfRenderer(RendererLike):"
    text = text.replace(anchor, inject, 1)

    old = """        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **kwargs,
        )

        # NOTE: use_unified_vision_chunk"""
    new = """        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **_gemma4_metax_chat_template_kwargs_for_safe_apply(kwargs),
        )
        prompt_raw = _maybe_gemma4_metax_align_chat_prompt_string(
            model_config=model_config,
            prompt_raw=prompt_raw,
            conversation=conversation,
        )

        # NOTE: use_unified_vision_chunk"""
    n = text.count(old)
    if n != 2:
        return f"vllm hf gemma4 metax chat align: inject sites {n} != 2 (unexpected vLLM layout)"
    text = text.replace(old, new, 2)

    bak = hf_py.with_suffix(hf_py.suffix + ".bak.gemma4_metax_chat_align")
    if hf_py.is_file() and not bak.exists():
        shutil.copy2(hf_py, bak)
    hf_py.write_text(text)
    return "vllm hf gemma4 metax chat align: applied"


def patch_vllm_gemma4_reasoning_parser_vendor(site_root: Path) -> str:
    """Replace site ``gemma4_reasoning_parser.py`` from repo ``vendor/vllm`` tree (if available)."""
    here = Path(__file__).resolve().parent
    src = here.parent / "vendor/vllm/vllm/reasoning/gemma4_reasoning_parser.py"
    dst = site_root / "reasoning/gemma4_reasoning_parser.py"
    if not src.is_file():
        return "gemma4 reasoning parser vendor copy: skip (no vendor src beside script)"
    if not dst.parent.is_dir():
        return "gemma4 reasoning parser vendor copy: skip (no reasoning/ dir)"
    marker = "def _extract_reasoning_from_token_ids"
    cur = dst.read_text() if dst.is_file() else ""
    if marker in cur:
        return "gemma4 reasoning parser vendor copy: already present"
    bak = dst.with_suffix(dst.suffix + ".bak.gemma4_token_extract")
    if dst.is_file() and not bak.exists():
        shutil.copy2(dst, bak)
    shutil.copy2(src, dst)
    return "gemma4 reasoning parser vendor copy: applied"


def patch_vllm_base_gemma4_attention_gate_widen(base_py: Path) -> str:
    """Upgrade trees patched with the legacy ``model_type == gemma4_text`` gate only."""
    text = base_py.read_text()
    if "_gemma4_alt" in text:
        return "vllm base gemma4 gate: already widened"
    narrow = """            tc_layer_types = getattr(text_config, "layer_types", None)
            if (
                getattr(text_config, "model_type", None) == "gemma4_text"
                and tc_layer_types is not None
                and i < len(tc_layer_types)
            ):"""
    wide = """            tc_layer_types = getattr(text_config, "layer_types", None)
            _text_mt = getattr(text_config, "model_type", None)
            _gemma4_alt = _text_mt in ("gemma4_text", "gemma4") and getattr(
                text_config, "global_head_dim", None
            ) is not None
            if (
                _gemma4_alt
                and tc_layer_types is not None
                and i < len(tc_layer_types)
            ):"""
    if narrow not in text:
        return "vllm base gemma4 gate: skip (no legacy narrow gate)"
    base_py.write_text(text.replace(narrow, wide, 1))
    return "vllm base gemma4 gate: widened"


def patch_metax_triton_gemma3_swa_collision(triton_py: Path) -> str:
    """Gemma4 sliding layers share (head_dim=128, window=1024) with Gemma3-27B; default off tile-32 fast path for 128."""
    text = triton_py.read_text()
    if "VLLM_METAX_GEMMA3_27B_TILE32" in text:
        return "metax triton swa gate: already applied"
    old = '''def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)
'''
    new = '''def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Gate the Gemma3 tile=32 fast path (see _get_tile_size).

    Gemma3 uses (head_size, sliding_window) pairs (128, 1024) for 27B and
    (256, 1024) for smaller checkpoints. Gemma4 interleaved layers use
    (128, 1024) for sliding windows as well, so the naive signature collides.
    On MACA, treating those Gemma4 layers as Gemma3-27B can yield wrong logits.

    Default: enable tile=32 only for (256, 1024). For Gemma3-27B on MetaX,
    set ``VLLM_METAX_GEMMA3_27B_TILE32=1``.
    """
    if sliding_window != 1024 or head_size not in (128, 256):
        return False
    if head_size == 256:
        return True
    return os.environ.get("VLLM_METAX_GEMMA3_27B_TILE32", "0") == "1"
'''
    if old not in text:
        return "metax triton swa gate: anchor missing (vendor already diverged?)"
    if "\nimport os\n" not in text and not text.startswith("import os"):
        text = text.replace("import torch\n", "import os\n\nimport torch\n", 1)
    text = text.replace(old, new, 1)
    triton_py.write_text(text)
    return "metax triton swa gate: applied"


def patch_metax_flex_attention_smem(flex_py: Path) -> str:
    """FLEX_ATTENTION on 64KiB smem: one halving can still exceed limit for large head_dim."""
    text = flex_py.read_text()
    if "64KiB-class devices" in text:
        return "metax flex smem: already patched"
    old = """            if max_shared_memory < 144 * 1024:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)"""
    new = """            if max_shared_memory < 144 * 1024:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )
            # 64KiB-class devices: one halving still overshoots Flex/Triton smem for
            # large head_dim (e.g. Gemma4 512); halve again while staying >= bound.
            if max_shared_memory <= 65536:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)"""
    if old not in text:
        return "metax flex smem: anchor missing (flex_attention diverged?)"
    bak = flex_py.with_suffix(".py.bak.flex_smem")
    if not bak.exists():
        bak.write_text(text)
    flex_py.write_text(text.replace(old, new, 1))
    return "metax flex smem: applied"


def patch_metax_flex_extra_halve_env(flex_py: Path) -> str:
    """Optional third BLOCK_M/N halving on 64KiB when VLLM_METAX_FLEX_EXTRA_HALVE=1."""
    text = flex_py.read_text()
    if "Optional third halving" in text:
        return "metax flex extra_halve: already patched"
    if "64KiB-class devices" not in text:
        return "metax flex extra_halve: skip (64KiB-class patch missing)"
    old = """            if max_shared_memory <= 65536:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)"""
    new = """            if max_shared_memory <= 65536:
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )
            # Optional third halving (still >= block_lower_bound): set
            # VLLM_METAX_FLEX_EXTRA_HALVE=1 when Flex still hits OutOfResources on 64KiB MACA.
            if max_shared_memory <= 65536 and os.environ.get(
                "VLLM_METAX_FLEX_EXTRA_HALVE", ""
            ).strip() in ("1", "true", "yes"):
                block_m_candidate = ensure_divisible(
                    max(1, block_m_candidate // 2), block_m
                )
                block_n_candidate = ensure_divisible(
                    max(1, block_n_candidate // 2), block_n
                )

        block_m_candidate = max(block_m_candidate, block_lower_bound)"""
    if old not in text:
        return "metax flex extra_halve: anchor missing (unexpected flex layout)"
    if "import os\n" not in text[:800]:
        if "import math\n" in text:
            text = text.replace("import math\n", "import math\nimport os\n", 1)
        else:
            return "metax flex extra_halve: anchor missing (import math)"
    flex_py.write_text(text.replace(old, new, 1))
    return "metax flex extra_halve: applied"


def patch_metax_flex_torch_eager_gate(flex_py: Path) -> str:
    """``torch.compile(flex_attention)`` can emit Triton kernels with smem > 65536 on MACA (Inductor)."""
    text = flex_py.read_text()
    if "VLLM_METAX_FLEX_EAGER" in text and "flex_attention_compiled = flex_attention" in text:
        return "metax flex eager gate: already patched"
    old = """torch._dynamo.config.recompile_limit = 16
create_block_mask_compiled = torch.compile(
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)"""
    new = """torch._dynamo.config.recompile_limit = 16
# VLLM_METAX_FLEX_EAGER=1: skip torch.compile (Inductor Triton smem often > 65536 on MACA for flex_attention).
if os.environ.get("VLLM_METAX_FLEX_EAGER", "").strip() in ("1", "true", "yes"):
    logger.info(
        "FLEX_ATTENTION: VLLM_METAX_FLEX_EAGER=1 — skipping torch.compile on "
        "flex_attention / create_block_mask (avoids Inductor smem OOR on 64KiB-class GPUs)."
    )
    create_block_mask_compiled = create_block_mask
    flex_attention_compiled = flex_attention
else:
    create_block_mask_compiled = torch.compile(
        create_block_mask, fullgraph=True, mode="reduce-overhead"
    )
    flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)"""
    if old not in text:
        return "metax flex eager gate: anchor missing (flex_attention diverged?)"
    flex_py.write_text(text.replace(old, new, 1))
    return "metax flex eager gate: applied"


def patch_metax_triton_gemma4_maca(triton_py: Path) -> str:
    """Gemma4 full layers (head_dim 512): TILE 16 and launch num_stages=1 for MACA 65536B smem cap.

    TILE must stay >= 16 (Triton tl.dot). BLOCK_M must stay >= 16. If still OOR, vendor kernel change
    or non-Triton attention path is needed.
    """
    original = triton_py.read_text()
    text = original
    changed: list[str] = []

    block_bad = """    # MACA Gemma4 full (head_dim 512): reduce BLOCK_M so TILE=16 fits in 64KiB smem (tl.dot needs TILE>=16)
    if head_size >= 448 and BLOCK_M == 16 and 8 % num_queries_per_kv == 0:
        BLOCK_M = 8
        BLOCK_Q = BLOCK_M // num_queries_per_kv

"""
    if block_bad in text:
        text = text.replace(block_bad, "", 1)
        changed.append("removed_invalid_block_m")

    tile_bad_return8 = """    # Gemma4 full attention (head_dim 512) on MACA: Triton shared memory limit 65536
    if head_size >= 448:
        return 8

"""
    # Decode cannot use TILE 8: Triton tl.dot requires the contracting dim >= 16
    # (see error: K shape [512, 8]).
    tile_body_old_decode8 = "    if head_size >= 448:\n        return 16 if is_prefill else 8\n"
    tile_body_ok = "    if head_size >= 448:\n        return 16\n"
    tile_ok = """    # Gemma4 full attention (head_dim 512) on MACA: TILE 16 prefill+decode (tl.dot needs >=16)
    if head_size >= 448:
        return 16

"""
    if tile_body_old_decode8 in text:
        text = text.replace(tile_body_old_decode8, tile_body_ok, 1)
        changed.append("tile_decode16")
    if tile_bad_return8 in text:
        text = text.replace(tile_bad_return8, tile_ok, 1)
        changed.append("tile_fix")
    elif tile_body_ok not in text:
        old = """    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32"""
        new = """    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Gemma4 full attention (head_dim 512) on MACA: TILE 16 prefill+decode (tl.dot needs >=16)
    if head_size >= 448:
        return 16

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32"""
        if old in text:
            text = text.replace(old, new, 1)
            changed.append("tile_apply")
        elif "head_size >= 448" in text and "def _get_tile_size(" in text:
            # Vendor / prior run already inserted Gemma4 full-head block (e.g. tile env); do not fail.
            pass
        else:
            return "metax triton gemma4: anchor missing (_get_tile_size)"

    # MACA: unified attention can request ~81920B shared memory with default pipelining;
    # hardware limit 65536. Force num_stages=1 on kernel launches (Triton run API).
    launch_2d_old = """            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )
    else:
        kernel_unified_attention_3d[
"""
    launch_2d_new = """            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            num_stages=1,
        )
    else:
        kernel_unified_attention_3d[
"""
    if launch_2d_old in text:
        text = text.replace(launch_2d_old, launch_2d_new, 1)
        changed.append("launch_2d_num_stages1")

    launch_3d_old = """            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
"""
    launch_3d_new = """            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            num_stages=1,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
"""
    if launch_3d_old in text:
        text = text.replace(launch_3d_old, launch_3d_new, 1)
        changed.append("launch_3d_num_stages1")

    launch_rs_old = """            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
"""
    launch_rs_new = """            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
            num_stages=1,
        )
"""
    if launch_rs_old in text:
        text = text.replace(launch_rs_old, launch_rs_new, 1)
        changed.append("launch_reduce_num_stages1")

    if not changed:
        return "metax triton gemma4: already patched"

    bak = triton_py.with_suffix(".py.bak.maca_gemma4")
    if not bak.exists():
        bak.write_text(original)
    triton_py.write_text(text)
    return "metax triton gemma4: " + ", ".join(changed)


def patch_metax_triton_gemma4_full_head_tile_env(triton_py: Path) -> str:
    """Gemma4 full-head layers: ``VLLM_METAX_GEMMA4_FULL_HEAD_TILE=16|32`` overrides TILE (see vendor)."""
    text = triton_py.read_text()
    if "VLLM_METAX_GEMMA4_FULL_HEAD_TILE" in text and "_tile = os.environ.get" in text:
        return "metax gemma4 full-head tile env: already patched"

    new_block = """    # Gemma4 full attention (head_dim 512) on MACA: TILE>=16 for tl.dot; smem via num_stages=1 on launch.
    # Optional A/B: ``VLLM_METAX_GEMMA4_FULL_HEAD_TILE=16`` or ``32`` (32 may exceed 64KiB smem on some MACA builds).
    if head_size >= 448:
        _tile = os.environ.get("VLLM_METAX_GEMMA4_FULL_HEAD_TILE", "").strip()
        if _tile in ("16", "32"):
            return int(_tile)
        return 16

    # Default behavior"""

    old_a = """    # Gemma4 full attention (head_dim 512) on MACA: TILE>=16 for tl.dot; smem via num_stages=1 on launch
    if head_size >= 448:
        return 16

    # Default behavior"""
    old_b = """    # Gemma4 full attention (head_dim 512) on MACA: TILE 16 prefill+decode (tl.dot needs >=16)
    if head_size >= 448:
        return 16

    # Default behavior"""
    for label, old in (("a", old_a), ("b", old_b)):
        if old in text:
            triton_py.write_text(text.replace(old, new_block, 1))
            return f"metax gemma4 full-head tile env: applied (variant {label})"
    return "metax gemma4 full-head tile env: anchor missing"


def patch_metax_triton_attn_gemma4_force_2d(triton_attn_py: Path) -> str:
    """Optional ``VLLM_METAX_GEMMA4_FORCE_2D=1`` — seq_threshold_3D=None -> 2D unified kernel."""
    if not triton_attn_py.is_file():
        return "metax triton_attn force2d: skip (missing)"
    original = triton_attn_py.read_text()
    text = original
    if "VLLM_METAX_GEMMA4_FORCE_2D" in text:
        return "metax triton_attn force2d: already patched"

    old_import = """from dataclasses import dataclass
from typing import ClassVar

import torch
"""
    new_import = """from dataclasses import dataclass
import os
from typing import ClassVar

import torch
"""
    if "import os\nfrom typing import ClassVar" not in text and old_import in text:
        text = text.replace(old_import, new_import, 1)

    old_attn = """        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
"""
    new_attn = """        seq_threshold_3D = attn_metadata.seq_threshold_3D
        # Gemma4 full-attention layers (head_dim 512): decode often uses the 3D segmented kernel path.
        # Optional MACA A/B: force the fused 2D unified kernel (``seq_threshold_3D is None`` triggers it).
        _hd = int(query.shape[2]) if query.ndim >= 3 else 0
        if _hd >= 448 and os.environ.get("VLLM_METAX_GEMMA4_FORCE_2D", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            seq_threshold_3D = None
            logger.info_once(
                "METAX Gemma4: VLLM_METAX_GEMMA4_FORCE_2D=1 — 2D unified attention (head_dim=%s).",
                _hd,
            )
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
"""
    if old_attn not in text:
        return "metax triton_attn force2d: anchor missing (triton_attn diverged?)"
    text = text.replace(old_attn, new_attn, 1)

    bak = triton_attn_py.with_suffix(".py.bak.gemma4_force2d")
    if not bak.exists():
        bak.write_text(original)
    triton_attn_py.write_text(text)
    return "metax triton_attn force2d: applied"


def main() -> None:
    root = _site()
    moe_py = root / "model_executor/models/transformers/moe.py"
    utils_py = root / "model_executor/models/utils.py"
    base_py = root / "model_executor/models/transformers/base.py"
    chat_serving_py = root / "entrypoints/openai/chat_completion/serving.py"
    resp_serving_py = root / "entrypoints/openai/responses/serving.py"
    for p in (moe_py, utils_py, base_py, chat_serving_py):
        if not p.is_file():
            print("missing:", p, file=sys.stderr)
            sys.exit(1)
    # Invalidate stale bytecode
    for pyc in moe_py.parent.glob("__pycache__/moe.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    for pyc in utils_py.parent.glob("__pycache__/utils.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    for pyc in base_py.parent.glob("__pycache__/base.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    for pyc in chat_serving_py.parent.glob("__pycache__/serving.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass

    print(patch_moe_topk(moe_py))
    print(patch_moe_skip_fused(moe_py))
    print(maybe_restore_moe_skip_fused(moe_py))
    print(patch_utils_buffers(utils_py))
    print(patch_vllm_base_gemma4_attention(base_py))
    print(patch_vllm_base_gemma4_attention_gate_widen(base_py))
    print(patch_vllm_base_gemma4_kv_gather_colwise(base_py))
    print(patch_vllm_chat_completion_reasoning_adjust(chat_serving_py))
    if resp_serving_py.is_file():
        for pyc in resp_serving_py.parent.glob("__pycache__/serving.*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass
        print(patch_vllm_responses_reasoning_adjust(resp_serving_py))
    else:
        print("vllm responses reasoning adjust: skip (serving.py missing)")

    print(patch_vllm_chat_completion_reasoning_extract_ids(chat_serving_py))
    if resp_serving_py.is_file():
        print(patch_vllm_responses_reasoning_extract_ids(resp_serving_py))
    else:
        print("vllm responses reasoning extract_ids: skip (serving.py missing)")
    parser_py = root / "entrypoints/openai/parser/responses_parser.py"
    if parser_py.is_file():
        for pyc in parser_py.parent.glob("__pycache__/responses_parser.*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass
        print(patch_vllm_responses_parser_reasoning_extract_ids(parser_py))
    else:
        print("vllm responses_parser reasoning extract_ids: skip (file missing)")
    print(patch_vllm_gemma4_reasoning_parser_vendor(root))

    hf_renderer_py = root / "renderers/hf.py"
    if hf_renderer_py.is_file():
        for pyc in hf_renderer_py.parent.glob("__pycache__/hf.*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass
        print(patch_vllm_hf_gemma4_metax_chat_prompt_align(hf_renderer_py))
    else:
        print("vllm hf gemma4 metax chat align: skip (renderers/hf.py missing)")

    import transformers

    tf_moe = Path(transformers.__file__).resolve().parent / "integrations/moe.py"
    if not tf_moe.is_file():
        print("missing:", tf_moe, file=sys.stderr)
        sys.exit(1)
    for pyc in tf_moe.parent.glob("__pycache__/moe.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    print(patch_transformers_moe_maca(tf_moe))

    try:
        import vllm_metax

        triton_py = (
            Path(vllm_metax.__file__).resolve().parent
            / "v1/attention/ops/triton_unified_attention.py"
        )
        if triton_py.is_file():
            for pyc in triton_py.parent.glob(
                "__pycache__/triton_unified_attention.*.pyc"
            ):
                try:
                    pyc.unlink()
                except OSError:
                    pass
            print(patch_metax_triton_gemma4_maca(triton_py))
            print(patch_metax_triton_gemma3_swa_collision(triton_py))
            print(patch_metax_triton_gemma4_full_head_tile_env(triton_py))
            triton_attn_py = (
                Path(vllm_metax.__file__).resolve().parent
                / "v1/attention/backends/triton_attn.py"
            )
            if triton_attn_py.is_file():
                for pyc in triton_attn_py.parent.glob("__pycache__/triton_attn.*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass
                print(patch_metax_triton_attn_gemma4_force_2d(triton_attn_py))
            else:
                print("metax triton_attn force2d: skip (triton_attn.py missing)")
            flex_py = (
                Path(vllm_metax.__file__).resolve().parent
                / "v1/attention/backends/flex_attention.py"
            )
            if flex_py.is_file():
                for pyc in flex_py.parent.glob("__pycache__/flex_attention.*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass
                print(patch_metax_flex_attention_smem(flex_py))
                print(patch_metax_flex_extra_halve_env(flex_py))
                print(patch_metax_flex_torch_eager_gate(flex_py))
            else:
                print("metax flex smem: skip (flex_attention.py missing)")
        else:
            print("metax triton tile: skip (triton_unified_attention.py missing)")
    except ImportError:
        print("metax triton tile: skip (vllm_metax not installed)")

    print("done. Restart vLLM.")


if __name__ == "__main__":
    main()
