#!/usr/bin/env python3
"""
vLLM entry bootstrap for Gemma4 on MetaX MACA (沐曦).

Must be used as the Python entrypoint *instead of* `python -m vllm.entrypoints.openai.api_server`
so patches run before vLLM imports the model stack.

What this does (runtime, no file edits):
  1) transformers: is_grouped_mm_available -> False (belt-and-suspenders)
  2) transformers AutoModel.from_config: inject experts_implementation for model_type==gemma4
     so MoE uses batched_mm (torch.bmm) or eager — not torch._grouped_mm / grouped_mm
  3) transformers.integrations.moe._grouped_mm: always use
     torch.ops.transformers.grouped_mm_fallback (MACA exposes _grouped_mm but sm90 path fails)
  4) Optional: torch._grouped_mm / F.grouped_mm shim -> same fallback when kwargs match

Disk patches (top_k_experts, skip FusedMoE for gate_up_proj, layer_scalar buffers):
  run once:  python3 workbench/gemma4-vllm/scripts/apply_vllm_site_patches.py

Recommended vLLM flags for first bring-up:
  --enforce-eager

Experts backend override (optional):
  GEMMA4_MACA_EXPERTS_BACKEND=batched_mm   (default, usually best on GPU)
  GEMMA4_MACA_EXPERTS_BACKEND=eager        (slowest, most compatible)

Example:
  python3 vllm_bootstrap_gemma4_maca.py \\
    --model /data/gemma-4-26B-A4B-it \\
    --served-model-name gemma-4-26B-A4B-it \\
    --max-model-len 4096 --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.85 \\
    --port 18001 --host 0.0.0.0 --enforce-eager
"""
from __future__ import annotations

import os
import runpy
import sys


def _backend() -> str:
    v = os.environ.get("GEMMA4_MACA_EXPERTS_BACKEND", "batched_mm").strip().lower()
    if v not in ("batched_mm", "eager"):
        return "batched_mm"
    return v


def _patch_transformers_grouped_mm_flag() -> None:
    import transformers.utils.import_utils as iu

    iu.is_grouped_mm_available = lambda: False
    if hasattr(iu.is_grouped_mm_available, "cache_clear"):
        try:
            iu.is_grouped_mm_available.cache_clear()
        except Exception:
            pass


def _patch_auto_model_from_config() -> None:
    import transformers.models.auto.modeling_auto as modeling_auto

    AM = modeling_auto.AutoModel
    _orig = AM.from_config.__func__
    backend = _backend()

    @classmethod
    def _from_config(cls, config, *args, **kwargs):
        if getattr(config, "model_type", None) == "gemma4":
            if "experts_implementation" not in kwargs:
                # Multimodal Gemma4: text MoE must not use grouped_mm on MACA
                kwargs["experts_implementation"] = {
                    "text_config": backend,
                    "vision_config": "eager",
                    "audio_config": "eager",
                }
        try:
            return _orig(cls, config, *args, **kwargs)
        except TypeError as e:
            # Some stacks only accept a plain string, not per-backbone dict
            if (
                getattr(config, "model_type", None) == "gemma4"
                and "experts_implementation" in str(e).lower()
            ):
                kw2 = dict(kwargs)
                kw2["experts_implementation"] = backend
                return _orig(cls, config, *args, **kw2)
            raise

    AM.from_config = _from_config


def _patch_moe_grouped_mm_force_fallback() -> None:
    """_can_use_grouped_mm() is True on MACA (op exists) but runtime requires sm90; force HF fallback."""
    import torch
    import transformers.integrations.moe as moe_mod

    def _grouped_mm_fallback_only(
        input: torch.Tensor,
        weight: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.transformers.grouped_mm_fallback(
            input.to(weight.dtype), weight, offs=offs
        )

    moe_mod._grouped_mm = _grouped_mm_fallback_only  # type: ignore[assignment]


def _patch_torch_grouped_mm_shim() -> None:
    """Direct torch._grouped_mm callers (bypassing moe._grouped_mm): use HF fallback."""
    import torch

    if not hasattr(torch, "_grouped_mm"):
        return

    def _shim(input, weight, *, offs):  # type: ignore[no-untyped-def]
        return torch.ops.transformers.grouped_mm_fallback(
            input.to(weight.dtype), weight, offs=offs
        )

    torch._grouped_mm = _shim  # type: ignore[misc, assignment]


def _apply_runtime_patches() -> None:
    _tile = os.environ.get("VLLM_METAX_GEMMA3_27B_TILE32", "0").strip().lower()
    if _tile in ("1", "true", "yes"):
        print(
            "GEMMA4_BOOTSTRAP_WARN: VLLM_METAX_GEMMA3_27B_TILE32=1 enables Gemma3-27B "
            "tile=32 for (head_dim=128, sliding=1024). Gemma4 SWA uses the same signature; "
            "wrong path => wrong logits. For Gemma4 keep 0.",
            file=sys.stderr,
            flush=True,
        )
    _patch_transformers_grouped_mm_flag()
    _patch_auto_model_from_config()
    _patch_moe_grouped_mm_force_fallback()
    _patch_torch_grouped_mm_shim()


def main() -> None:
    _apply_runtime_patches()
    # Execute vLLM OpenAI server as if: python -m vllm.entrypoints.openai.api_server ...
    runpy.run_module(
        "vllm.entrypoints.openai.api_server",
        run_name="__main__",
        alter_sys=True,
    )


if __name__ == "__main__":
    main()
