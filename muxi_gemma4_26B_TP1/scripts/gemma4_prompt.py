# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Google / HF Gemma4 chat_template.jinja (逻辑对齐单轮 user → model 生成前缀)
"""单轮 Gemma4-IT 文本前缀（用于 vLLM ``/v1/completions``，绕过 OpenAI chat 栈对模板的二次处理差异）。

与 ``google/gemma-4-26B-A4B-it`` 的 ``chat_template.jinja`` 一致路径：
- 无 system / tools、首条为 user、``add_generation_prompt=True``
- ``enable_thinking=False`` 时：在 ``<|turn>model`` 后追加 ``<|channel>thought\\n ``（与模板一致）
"""
from __future__ import annotations


def single_turn_metax_no_bos_completions_prefix(user_text: str) -> str:
    """沐曦现场可对齐 HF 的一条 ``/v1/completions`` 前缀（无字面 ``<bos>``，配合 ``add_special_tokens=true``）。"""
    u = (user_text or "").strip()
    return f"<|turn>user\n{u} \n<|turn>model\n<|channel>thought\n "


def single_turn_completion_prefix(user_text: str, *, enable_thinking: bool = False) -> str:
    u = (user_text or "").strip()
    # 与多数 Gemma 权重解码习惯一致：显式 BOS，避免首 token 漂移成乱码
    bos = "<bos>"
    # jinja: <|turn>user\n + content + " \n"
    out = f"{bos}<|turn>user\n{u} \n<|turn>model\n"
    if not enable_thinking:
        out += "<|channel>thought\n "
    return out
