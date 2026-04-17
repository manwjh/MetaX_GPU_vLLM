# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 chat 输出客户端拆解（不依赖 vLLM --reasoning-parser）。

逻辑与上游 ``vllm.reasoning.gemma4_utils.parse_thinking_output`` 对齐，见仓库：
``docs/其他资料/upstream-snapshots/vllm-main-reasoning/gemma4_utils.py``。
"""

from __future__ import annotations

_THINKING_START_TAG = "<|channel>"
_THINKING_END_TAG = "<channel|>"
_TURN_END_TAG = "<turn|>"


def _strip_leading_channel_roles(text: str) -> str:
    """Peel repeated ``<|channel>role`` headers (e.g. ``final``) left in the answer tail."""
    s = text.strip()
    while s.startswith(_THINKING_START_TAG):
        rest = s[len(_THINKING_START_TAG) :]
        if "\n" not in rest:
            break
        _role, _nl, remainder = rest.partition("\n")
        s = remainder.lstrip()
    return s


def _strip_thought_label(text: str) -> str:
    if text.startswith("thought\n"):
        return text[len("thought\n") :]
    return text


def _clean_answer(text: str) -> str:
    text = text.strip()
    if text.endswith(_TURN_END_TAG):
        text = text[: -len(_TURN_END_TAG)].rstrip()
    if text.endswith("<eos>"):
        text = text[:-5].rstrip()
    return text


def _peel_inline_thinking_after_first_close(answer: str, thinking_parts: list[str]) -> str:
    """If the answer tail still contains ``<|channel>…<channel|>`` blocks, peel them."""
    a = answer.strip()
    while _THINKING_START_TAG in a and _THINKING_END_TAG in a:
        before, sep, after = a.partition(_THINKING_START_TAG)
        if not sep:
            break
        mid, _, rest = after.partition(_THINKING_END_TAG)
        frag = _strip_thought_label(mid.strip()).strip()
        if frag:
            thinking_parts.append(frag)
        a = (before + rest).strip()
    return a


def parse_thinking_output(text: str) -> dict[str, str | None]:
    thinking_parts: list[str] = []
    if _THINKING_END_TAG in text:
        parts = text.split(_THINKING_END_TAG, 1)
        thinking_block = parts[0]
        answer = _clean_answer(parts[1])
        if _THINKING_START_TAG in thinking_block:
            thinking = thinking_block.split(_THINKING_START_TAG, 1)[1]
        else:
            thinking = thinking_block
        thinking = _strip_thought_label(thinking.strip()).strip()
        if thinking:
            thinking_parts.append(thinking)
        answer = _peel_inline_thinking_after_first_close(answer, thinking_parts)
        answer = _strip_leading_channel_roles(_clean_answer(answer))
        merged = "\n".join(thinking_parts) if thinking_parts else None
        return {"thinking": merged, "answer": answer}
    answer = _strip_thought_label(text)
    answer = _clean_answer(_strip_leading_channel_roles(answer))
    return {"thinking": None, "answer": answer}


if __name__ == "__main__":
    r1 = parse_thinking_output("<|channel>thought\nhello<channel|>长沙。")
    assert r1["thinking"] == "hello", r1
    assert r1["answer"] == "长沙。", r1
    r2 = parse_thinking_output("<|channel>thought\nx<channel|><|channel>final\n答")
    assert r2["answer"] == "答", r2
    r3 = parse_thinking_output("前缀<|channel>thought\nx<channel|>长沙。")
    assert "长沙" in (r3["answer"] or ""), r3
    print("gemma4_client_parse: ok")
