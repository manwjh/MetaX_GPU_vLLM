#!/usr/bin/env python3
"""OpenAI-compatible chat smoke test against vLLM; ends with a human-readable report.

Gemma4 无服务端 reasoning-parser 时可用 ``--parse-gemma4``，逻辑见 ``gemma4_client_parse.py``。

路径：请在**仓库根目录**执行 ``python3 scripts/muxi_gemma4/test_vllm_chat.py``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_local() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _load_parse():
    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from gemma4_client_parse import parse_thinking_output

    return parse_thinking_output


def _truncate(s: str | None, max_len: int) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _print_report(lines: list[str], title: str = "vLLM 烟测报告") -> None:
    sep = "=" * 72
    print()
    print(sep)
    print(title)
    print(sep)
    for line in lines:
        print(line)


def main() -> int:
    p = argparse.ArgumentParser(
        description="POST OpenAI-compatible vLLM API; print answer + report.",
    )
    p.add_argument(
        "--api",
        choices=("chat", "completions"),
        default="chat",
        help="chat=/v1/chat/completions；completions=/v1/completions（Gemma4 单轮前缀见 gemma4_prompt）",
    )
    p.add_argument(
        "--url",
        default="",
        help="完整 POST URL；留空则按 --api 默认 http://127.0.0.1:18001/v1/...",
    )
    p.add_argument(
        "--model",
        default="gemma-4-26B-A4B-it",
        help="model id (must match --served-model-name on server)",
    )
    p.add_argument("--prompt", default="用一句话介绍长沙。")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy; Gemma4 烟测建议先 0",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="pass chat_template_kwargs.enable_thinking=true",
    )
    p.add_argument(
        "--parse-gemma4",
        action="store_true",
        help="post-process message.content (Gemma4 <|channel> / <channel|>)",
    )
    p.add_argument(
        "--vllm-default-decode",
        action="store_true",
        help="不覆盖 vLLM 默认：仍用 skip_special_tokens=true（会剥掉 <|channel> 字面值，Gemma4 客户端拆解会失效）",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--no-report",
        action="store_true",
        help="only print assistant text (no trailing report block)",
    )
    p.add_argument(
        "--report-json",
        action="store_true",
        help="print one JSON object with metrics + body fields instead of text report",
    )
    args = p.parse_args()

    if not (args.url or "").strip():
        base = "http://127.0.0.1:18001"
        args.url = f"{base}/v1/completions" if args.api == "completions" else f"{base}/v1/chat/completions"
    url = (args.url or "").strip()

    raw_prompt_for_report: str | None = None
    if args.api == "completions":
        _root = Path(__file__).resolve().parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from gemma4_prompt import single_turn_completion_prefix

        raw_prompt_for_report = single_turn_completion_prefix(
            args.prompt, enable_thinking=args.enable_thinking
        )
        body = {
            "model": args.model,
            "prompt": raw_prompt_for_report,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        # vLLM 默认 skip_special_tokens=True 会把 <|channel> / <channel|> 从 completion 文本里剥掉，
        # 客户端无法按 Gemma4 通道拆分（上游 gemma4_reasoning_parser 同样要求 false）。
        if not args.vllm_default_decode:
            body["skip_special_tokens"] = False
            # 前缀已含 HF 模板里的 <bos>，勿再让 tokenizer  prepend 一次 BOS。
            body["add_special_tokens"] = False
    else:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    if args.temperature <= 0:
        body["top_p"] = 1.0
    if args.api == "chat" and args.enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if args.api == "chat" and args.parse_gemma4 and not args.vllm_default_decode:
        body["skip_special_tokens"] = False

    payload = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_wall_start = _iso_local()
    t0 = time.perf_counter()
    out: dict[str, Any] | None = None
    http_status: int | None = None
    err_type: str | None = None
    err_detail: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            http_status = getattr(r, "status", None) or r.getcode()
            raw = r.read().decode()
            out = json.loads(raw)
    except urllib.error.HTTPError as e:
        http_status = e.code
        err_type = "HTTPError"
        err_detail = e.read().decode(errors="replace")[:8000]
    except urllib.error.URLError as e:
        err_type = "URLError"
        err_detail = str(e.reason) if getattr(e, "reason", None) else str(e)
    except json.JSONDecodeError as e:
        err_type = "JSONDecodeError"
        err_detail = str(e)
    except Exception as e:  # noqa: BLE001
        err_type = type(e).__name__
        err_detail = f"{e}\n{traceback.format_exc()}"[:8000]

    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000, 2)
    t_wall_end = _iso_local()

    if err_type:
        report_lines = [
            f"状态: 失败 ({err_type})",
            f"开始时间: {t_wall_start}",
            f"结束时间: {t_wall_end}",
            f"耗时: {latency_ms} ms（perf_counter）",
            f"URL: {url}",
            f"api: {args.api}",
            f"模型: {args.model}",
            f"max_tokens: {args.max_tokens}  temperature: {args.temperature}  timeout: {args.timeout}s",
            f"enable_thinking: {args.enable_thinking}  parse_gemma4: {args.parse_gemma4}  vllm_default_decode: {args.vllm_default_decode}",
            f"HTTP 状态: {http_status}",
            "",
            "[错误详情]",
            err_detail or "(空)",
        ]
        if args.report_json:
            obj = {
                "ok": False,
                "error": {"type": err_type, "detail": err_detail},
                "timing": {
                    "wall_start": t_wall_start,
                    "wall_end": t_wall_end,
                    "latency_ms": latency_ms,
                },
                "request": {
                    "url": url,
                    "api": args.api,
                    "model": args.model,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "timeout": args.timeout,
                    "enable_thinking": args.enable_thinking,
                    "parse_gemma4": args.parse_gemma4,
                    "vllm_default_decode": args.vllm_default_decode,
                },
                "http_status": http_status,
            }
            print(json.dumps(obj, ensure_ascii=False, indent=2))
        else:
            if not args.no_report:
                _print_report(report_lines)
        return 1

    assert out is not None

    if "choices" not in out:
        report_lines = [
            "状态: 失败（响应无 choices）",
            f"开始时间: {t_wall_start}",
            f"结束时间: {t_wall_end}",
            f"耗时: {latency_ms} ms",
            f"HTTP 状态: {http_status}",
            "",
            "[原始 JSON 摘要]",
            _truncate(json.dumps(out, ensure_ascii=False), 4000),
        ]
        if args.report_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {"type": "bad_response", "detail": "missing choices"},
                        "timing": {
                            "wall_start": t_wall_start,
                            "wall_end": t_wall_end,
                            "latency_ms": latency_ms,
                        },
                        "response_preview": out,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            if not args.no_report:
                _print_report(report_lines)
        return 1

    choice0 = out["choices"][0]
    if args.api == "completions":
        text = choice0.get("text") or ""
        rc = None
        finish = choice0.get("finish_reason")
    else:
        msg = choice0.get("message") or {}
        text = msg.get("content") or ""
        rc = msg.get("reasoning_content")
        finish = choice0.get("finish_reason")
    usage = out.get("usage") or {}

    thinking_out: str | None = None
    answer_out = text
    if args.parse_gemma4 and not rc:
        parse_thinking_output = _load_parse()
        parsed = parse_thinking_output(text)
        thinking_out = parsed["thinking"]
        answer_out = parsed["answer"] or ""

    report_title = "vLLM Completions 烟测报告" if args.api == "completions" else "vLLM Chat 烟测报告"
    prompt_note = (
        f"  用户句长度: {len(args.prompt)} 字符；completions 前缀总长: {len(raw_prompt_for_report or '')} 字符"
        if args.api == "completions"
        else f"  用户消息长度: {len(args.prompt)} 字符"
    )
    report_lines = [
        "状态: 成功",
        f"开始时间: {t_wall_start}",
        f"结束时间: {t_wall_end}",
        f"请求耗时: {latency_ms} ms（perf_counter，含读 body）",
        "",
        "[请求]",
        f"  URL: {url}",
        f"  api: {args.api}",
        f"  model: {args.model}",
        f"  max_tokens: {args.max_tokens}  temperature: {args.temperature}  timeout: {args.timeout}s",
        f"  enable_thinking: {args.enable_thinking}",
        f"  parse_gemma4: {args.parse_gemma4}  vllm_default_decode: {args.vllm_default_decode}",
        (
            "  vLLM decode: skip_special_tokens="
            + str(body.get("skip_special_tokens", "(未设置)"))
            + "  add_special_tokens="
            + str(body.get("add_special_tokens", "(未设置)"))
            if args.api == "completions"
            else "  vLLM decode: skip_special_tokens=" + str(body.get("skip_special_tokens", "(未设置)"))
        ),
        prompt_note,
        f"  POST body 字节: {len(payload)}",
        "",
        "[响应]",
        f"  HTTP: {http_status}",
        f"  id: {out.get('id', '(无)')}",
        f"  object: {out.get('object', '(无)')}",
        f"  finish_reason: {finish}",
        f"  reasoning_content: {'有' if rc else '无'}",
        "",
        "[usage]",
        f"  {json.dumps(usage, ensure_ascii=False) if usage else '(无)'}",
        "",
        "[助手输出长度]",
        f"  {'completion' if args.api == 'completions' else 'content'}: {len(text)} 字符",
    ]
    if rc:
        report_lines.append(f"  reasoning_content: {len(rc)} 字符")
    if thinking_out is not None:
        report_lines.extend(
            [
                f"  客户端解析 thinking: {len(thinking_out)} 字符",
                f"  客户端解析 answer: {len(answer_out)} 字符",
                "",
                "[thinking 预览（前 400 字）]",
                _truncate(thinking_out, 400) or "(无)",
            ]
        )
    elif (
        args.parse_gemma4
        and not rc
        and thinking_out is None
        and "<channel|>" not in text
        and "<|channel>" not in text
    ):
        tip = (
            "  未在输出中发现 <|channel> / <channel|>，客户端无法拆分 thought。"
            " 若当前为 chat，可试 --api completions；若已为 completions 仍异常，多为引擎/权重/解码路径问题，需沐曦镜像或 vLLM 版本排查。"
        )
        report_lines.extend(["", "[提示]", tip])
    report_lines.extend(
        [
            "",
            "[answer 预览（前 600 字）]",
            _truncate(answer_out if (args.parse_gemma4 and not rc) else text, 600)
            or "(空)",
        ]
    )

    if args.report_json:
        j: dict[str, Any] = {
            "ok": True,
            "timing": {
                "wall_start": t_wall_start,
                "wall_end": t_wall_end,
                "latency_ms": latency_ms,
            },
            "request": {
                "url": url,
                "api": args.api,
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "timeout": args.timeout,
                "enable_thinking": args.enable_thinking,
                "parse_gemma4": args.parse_gemma4,
                "vllm_default_decode": args.vllm_default_decode,
            },
            "http_status": http_status,
            "response_meta": {
                "id": out.get("id"),
                "object": out.get("object"),
                "finish_reason": finish,
                "usage": usage,
            },
            "message": (
                {"content": text, "reasoning_content": rc}
                if args.api == "chat"
                else {"text": text}
            ),
        }
        if raw_prompt_for_report is not None:
            j["request"]["completion_prompt_prefix"] = raw_prompt_for_report
        if thinking_out is not None:
            j["gemma4_client_parse"] = {
                "thinking": thinking_out,
                "answer": answer_out,
            }
        print(json.dumps(j, ensure_ascii=False, indent=2))
        return 0

    # 主输出：便于管道只取「正文」一行块（parse 时为拆解后的 answer）
    if args.parse_gemma4 and not rc:
        print(answer_out)
    elif rc:
        print(text)
        print()
        print("[reasoning_content]")
        print(rc)
    else:
        print(text)

    if not args.no_report:
        _print_report(report_lines, title=report_title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
