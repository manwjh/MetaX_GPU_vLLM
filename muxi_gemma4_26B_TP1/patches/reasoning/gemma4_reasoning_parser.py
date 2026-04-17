# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Role label that Gemma4 emits at the start of the thinking channel.
# The model generates: <|channel>thought\n...reasoning...<channel|>
# This prefix must be stripped to expose only the actual reasoning content.
_THOUGHT_PREFIX = "thought\n"


def _merge_reasoning_parts(a: str | None, b: str | None) -> str | None:
    parts: list[str] = []
    for x in (a, b):
        if not x:
            continue
        t = x.strip()
        if t:
            parts.append(t)
    if not parts:
        return None
    return "\n".join(parts)


def _join_reasoning_segments(segments: list[str]) -> str | None:
    """Merge one or more channel bodies; strip leading ``thought\\n`` per segment."""
    parts: list[str] = []
    for seg in segments:
        t = _strip_thought_label(seg.strip())
        if t:
            parts.append(t)
    if not parts:
        return None
    return "\n".join(parts)


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning/thinking
    content within its output. Thinking mode is activated by passing
    ``enable_thinking=True`` in the chat template kwargs, which injects a
    system turn containing <|think|> (token 98) to trigger chain-of-thought
    reasoning.

    Output pattern when thinking is enabled::

        <|channel>thought
        ...chain of thought reasoning...<channel|>
        Final answer text here.

    The ``thought\\n`` role label inside the channel delimiters is a
    structural artefact (analogous to ``user\\n`` in ``<|turn>user\\n...``).
    This parser strips it so that downstream consumers see only the
    actual reasoning text, consistent with the offline parser
    (``vllm.reasoning.gemma4_utils._strip_thought_label``).
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Instance state for streaming prefix stripping.
        # Tracks only the reasoning text received from the base parser,
        # independent of current_text (which may contain pre-reasoning
        # content and lacks special token text due to
        # skip_special_tokens=True).
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        """Preserve channel delimiter strings in decoded assistant text.

        ``skip_special_tokens=False`` keeps ``<|channel>`` / ``<channel|>`` in
        the stream. For ``PreTrainedTokenizerFast``, vLLM's incremental
        detokenizer sets ``spaces_between_special_tokens`` to True whenever
        that flag is left at its default, which can insert spaces *between*
        added-token strings and break exact substring matches for delimiters.
        Force it off when the request object exposes the field (Chat API).
        """
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<|channel>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        # Search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                # We're generating a tool call, so reasoning must be ended.
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                # We found a new turn or tool response token so don't consider
                # reasoning ended yet, since the model starts new reasoning
                # after these tokens.
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
        output_token_ids: Sequence[int] | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\n`` role label.

        When ``output_token_ids`` is provided (non-streaming chat path), we
        ``decode(..., skip_special_tokens=False)`` the **entire** completion and
        split on the literal delimiter strings. This matches HF-style handling
        where ``<channel|>`` may not appear as a single vocabulary id in
        ``output_token_ids`` even though the decoded text contains the marker.
        """
        if output_token_ids is not None and len(output_token_ids) > 0:
            by_ids = self._extract_reasoning_from_token_ids(output_token_ids)
            if by_ids is not None:
                return self._strip_nested_channel_from_content(*by_ids)

        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        pre0, sep0, tail0 = model_output.partition(self.start_token)
        if not sep0:
            return None, model_output
        r, c = self._extract_reasoning_string_multi(pre0, tail0)
        return self._strip_nested_channel_from_content(r, c)

    def _decode_completion_ids(self, ids: Sequence[int]) -> str:
        chunk = list(ids)
        if not chunk:
            return ""
        return self.model_tokenizer.decode(chunk, skip_special_tokens=False)

    def _strip_nested_channel_from_content(
        self, reasoning: str | None, content: str | None
    ) -> tuple[str | None, str | None]:
        """Move ``<|channel>…<channel|>`` fragments that leaked into ``content`` into reasoning."""
        if not content or self.start_token not in content:
            return reasoning, content
        r_acc = reasoning
        c = content
        for _ in range(64):
            if not c or self.start_token not in c:
                break
            if self.end_token not in c:
                break
            before, sep, tail = c.partition(self.start_token)
            if not sep:
                break
            r2, c2 = self._extract_reasoning_string_multi("", tail)
            r_acc = _merge_reasoning_parts(r_acc, r2)
            c = (before + (c2 or "")).strip()
        return r_acc, (c.strip() or None)

    def _extract_reasoning_string_multi(
        self, pre: str, after_first_start: str
    ) -> tuple[str | None, str | None]:
        """Consume ``(optional extra <|channel>)* body <channel|>`` segments; tail is content."""
        s, e = self.start_token, self.end_token
        chunks: list[str] = []
        cur = after_first_start
        while True:
            while cur.startswith(s):
                cur = cur[len(s) :]
            if not cur:
                return _join_reasoning_segments(chunks), (pre.strip() or None)
            if e not in cur:
                chunks.append(cur)
                return _join_reasoning_segments(chunks), (pre.strip() or None)
            mid, _, rest = cur.partition(e)
            chunks.append(mid)
            cur = rest
            nxt = cur.lstrip()
            if nxt.startswith(s):
                cur = nxt
                continue
            merged = (pre + cur).strip() or None
            return _join_reasoning_segments(chunks), merged

    def _extract_reasoning_from_token_ids(
        self, output_token_ids: Sequence[int]
    ) -> tuple[str | None, str | None] | None:
        ids = list(output_token_ids)
        if not ids:
            return None
        full = self._decode_completion_ids(ids)
        if self.start_token not in full and self.end_token not in full:
            return None
        pre0, sep0, tail0 = full.partition(self.start_token)
        if not sep0:
            return None
        return self._extract_reasoning_string_multi(pre0, tail0)

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract streaming reasoning, stripping ``thought\\n`` from the
        first reasoning delta(s).

        The ``thought\\n`` prefix may arrive as a single delta or split
        across multiple deltas (e.g. ``"thought"`` then ``"\\n"``). We
        buffer early reasoning tokens until we can determine whether the
        prefix is present, then emit the buffered content minus the
        prefix.

        Unlike the previous implementation which reconstructed accumulated
        reasoning from ``current_text``, this uses instance state
        (``_reasoning_text``) to track only the reasoning content returned
        by the base parser. This is necessary because
        ``skip_special_tokens=True`` (the vLLM default) causes the
        ``<|channel>`` delimiter to be invisible in ``current_text``,
        making it impossible to separate pre-reasoning content from
        reasoning content via string matching.
        """
        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if result is None:
            return None

        if result.reasoning is None:
            return result

        # Accumulate ONLY the reasoning text from base parser results.
        # This is immune to pre-reasoning content pollution.
        self._reasoning_text += result.reasoning

        # Once the prefix has been handled, all subsequent reasoning
        # deltas pass through unchanged.
        if self._prefix_stripped:
            return result

        # ---- Prefix stripping logic ----

        # Case 1: We've accumulated enough to confirm the prefix is
        # present. Strip it and pass through the remainder.
        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            # How much reasoning was accumulated before this delta?
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                # Prefix was already consumed by prior deltas; this
                # delta is entirely real content — pass through.
                self._prefix_stripped = True
                return result
            else:
                # Part or all of the prefix is in this delta.
                chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
                stripped = result.reasoning[chars_of_prefix_in_delta:]
                if stripped:
                    self._prefix_stripped = True
                    result.reasoning = stripped
                    return result
                else:
                    if len(self._reasoning_text) >= prefix_len:
                        self._prefix_stripped = True
                        result.reasoning = ""
                        return result
                    return None

        # Case 2: Accumulated text is a strict prefix of
        # _THOUGHT_PREFIX (e.g. we've only seen "thou" so far).
        # Buffer by suppressing — we can't yet tell if this will
        # become the full prefix or diverge.
        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        # Case 3: Accumulated text doesn't match the thought prefix
        # at all. This means prior deltas were buffered (suppressed
        # by Case 2) but the text diverged. Re-emit the full
        # accumulated text to avoid data loss.
        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
