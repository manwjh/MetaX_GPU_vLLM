#!/usr/bin/env python3
"""Terminal UI app for testing LLM services.

Current provider:
- openai_compatible

Architecture keeps adapters isolated so more providers can be added later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    # Better CJK/IME line editing behavior in many terminals.
    from prompt_toolkit import prompt as _pt_prompt
except Exception:  # noqa: BLE001
    _pt_prompt = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = APP_DIR / "profile.json"
DEFAULT_ENV_FILE = APP_DIR / ".env"
DEFAULT_ENV_EXAMPLE_FILE = APP_DIR / ".env.example"


@dataclass
class AppConfig:
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:18010"
    model: str = "gemma-4-26B-A4B-it"
    api_key: str = ""
    timeout_sec: float = 120.0


class ServiceAdapter:
    name = "base"

    def chat(
        self,
        cfg: AppConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        on_stream_text: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def health(self, cfg: AppConfig) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleAdapter(ServiceAdapter):
    name = "openai_compatible"

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        return headers

    @staticmethod
    def _extract_stream_text(evt: dict[str, Any]) -> tuple[str, str | None]:
        """Extract text from a stream event.

        Some servers emit incremental deltas in ``delta.content`` while others
        emit cumulative text in ``message.content``/``text``. We handle both.
        """
        choices = evt.get("choices") or []
        if not choices:
            return "", None
        c0 = choices[0] or {}
        finish_reason = c0.get("finish_reason")
        delta = c0.get("delta") or {}
        if isinstance(delta, dict):
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                return piece, finish_reason

        # Fallbacks for non-standard/cumulative streaming formats.
        msg = c0.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content, finish_reason
        text = c0.get("text")
        if isinstance(text, str) and text:
            return text, finish_reason
        return "", finish_reason

    def chat(
        self,
        cfg: AppConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        on_stream_text: Any | None = None,
    ) -> dict[str, Any]:
        url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stream:
            body["stream"] = True
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers=self._headers(cfg.api_key),
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            status = resp.getcode()
            if stream:
                answer_parts: list[str] = []
                assembled = ""
                finish_reason = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    piece, event_finish_reason = self._extract_stream_text(evt)
                    if piece:
                        # If server sends cumulative text, only emit new suffix.
                        if piece.startswith(assembled):
                            emit = piece[len(assembled) :]
                        # If server sends incremental text, emit as-is.
                        else:
                            emit = piece
                        if emit:
                            answer_parts.append(emit)
                            assembled += emit
                            if on_stream_text:
                                on_stream_text(emit)
                    finish_reason = event_finish_reason or finish_reason
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return {
                    "http_status": status,
                    "latency_ms": latency_ms,
                    "answer": "".join(answer_parts),
                    "usage": {},
                    "finish_reason": finish_reason,
                    "raw": {},
                }
            raw = resp.read().decode("utf-8", errors="replace")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        out = json.loads(raw)
        choices = out.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        return {
            "http_status": status,
            "latency_ms": latency_ms,
            "answer": message.get("content") or "",
            "usage": out.get("usage") or {},
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "raw": out,
        }

    def health(self, cfg: AppConfig) -> dict[str, Any]:
        url = f"{cfg.base_url.rstrip('/')}/v1/models"
        req = urllib.request.Request(url, method="GET", headers=self._headers(cfg.api_key))
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            status = resp.getcode()
            raw = resp.read().decode("utf-8", errors="replace")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        out = json.loads(raw)
        models = [item.get("id", "") for item in (out.get("data") or [])]
        return {"http_status": status, "latency_ms": latency_ms, "models": models, "raw": out}


ADAPTERS: dict[str, ServiceAdapter] = {
    OpenAICompatibleAdapter.name: OpenAICompatibleAdapter(),
}


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from .env into process environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def load_env_with_fallback(primary: Path, fallback: Path) -> Path | None:
    """Load env from primary, fallback to example when primary missing."""
    if primary.exists():
        load_env_file(primary)
        return primary
    if fallback.exists():
        load_env_file(fallback)
        return fallback
    return None


def save_profile(cfg: AppConfig, path: Path = DEFAULT_PROFILE) -> None:
    path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile(path: Path = DEFAULT_PROFILE) -> AppConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig(**data)


def _input_default(prompt: str, default: str) -> str:
    raw = _read_line(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _read_line(prompt_text: str) -> str:
    if _pt_prompt is not None:
        try:
            return _pt_prompt(prompt_text)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:  # noqa: BLE001
            # Fallback to builtin input when prompt_toolkit fails unexpectedly.
            pass
    return input(prompt_text)


def _input_api_key(current: str) -> str:
    hint = "***" if current else "(empty)"
    print("api_key input: Enter=keep, '-'=clear, or paste new key.")
    raw = _read_line(f"api_key [{hint}]: ").strip()
    if not raw:
        return current
    if raw == "-":
        return ""
    return raw


def _input_int(prompt: str, default: int) -> int:
    while True:
        raw = _read_line(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"invalid integer: {raw!r}")


def _input_float(prompt: str, default: float) -> float:
    while True:
        raw = _read_line(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"invalid number: {raw!r}")


def _print_error(prefix: str, exc: Exception) -> None:
    if isinstance(exc, urllib.error.HTTPError):
        detail = (exc.read() or b"").decode("utf-8", errors="replace")[:400]
        print(f"{prefix}: HTTP {exc.code} {detail}")
        return
    print(f"{prefix}: {type(exc).__name__}: {exc}")


def run_chat_once(
    cfg: AppConfig,
    prompt: str,
    max_tokens: int,
    temperature: float,
    as_json: bool,
    stream: bool = False,
) -> int:
    adapter = ADAPTERS[cfg.provider]
    try:
        if stream and not as_json:
            print("assistant> ", end="", flush=True)
            result = adapter.chat(
                cfg,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                on_stream_text=lambda s: print(s, end="", flush=True),
            )
            print()
        else:
            result = adapter.chat(
                cfg,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
    except Exception as exc:  # noqa: BLE001
        _print_error("chat failed", exc)
        return 1
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not stream:
        print("answer:", result["answer"])
    print("http_status:", result["http_status"], "latency_ms:", result["latency_ms"])
    print("usage:", json.dumps(result["usage"], ensure_ascii=False))
    return 0


def _run_chat_session(cfg: AppConfig, stream: bool = False) -> None:
    print("chat mode started. input '/exit' to quit this mode.")
    print("stream:", "on" if stream else "off")
    print(f"target: provider={cfg.provider} base_url={cfg.base_url} model={cfg.model}")
    try:
        health = ADAPTERS[cfg.provider].health(cfg)
        print(
            "health:",
            f"http_status={health['http_status']}",
            f"latency_ms={health['latency_ms']}",
            f"models={len(health['models'])}",
        )
    except Exception as exc:  # noqa: BLE001
        _print_error("health failed before chat", exc)
        print("tip: set base_url/model via .env or menu before chatting.")
    max_tokens = _input_int("max_tokens", 128)
    temperature = _input_float("temperature", 0.0)
    while True:
        prompt = _read_line("you> ").strip()
        if prompt == "/exit":
            break
        if not prompt:
            continue
        run_chat_once(
            cfg,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            as_json=False,
            stream=stream,
        )


def tui_loop(cfg: AppConfig) -> int:
    print("LLM TUI started. Type menu number and Enter.")
    try:
        while True:
            print("\n=== LLM TUI ===")
            print("1) Show config")
            print("2) Set provider")
            print("3) Set base_url")
            print("4) Set model")
            print("5) Set api_key")
            print("6) Health check (/v1/models)")
            print("7) Single prompt test")
            print("8) Chat session mode")
            print("9) Save profile")
            print("10) Load profile")
            print("0) Quit")
            choice = _read_line("> ").strip()

            if choice == "0":
                return 0
            if choice == "1":
                shown = asdict(cfg).copy()
                shown["api_key"] = "***" if shown["api_key"] else ""
                print(json.dumps(shown, ensure_ascii=False, indent=2))
            elif choice == "2":
                print("providers:", ", ".join(sorted(ADAPTERS.keys())))
                val = _input_default("provider", cfg.provider)
                if val not in ADAPTERS:
                    print("unknown provider")
                else:
                    cfg.provider = val
            elif choice == "3":
                cfg.base_url = _input_default("base_url", cfg.base_url)
            elif choice == "4":
                cfg.model = _input_default("model", cfg.model)
            elif choice == "5":
                cfg.api_key = _input_api_key(cfg.api_key)
            elif choice == "6":
                try:
                    result = ADAPTERS[cfg.provider].health(cfg)
                    print("http_status:", result["http_status"], "latency_ms:", result["latency_ms"])
                    print("models:", result["models"])
                except Exception as exc:  # noqa: BLE001
                    _print_error("health failed", exc)
            elif choice == "7":
                prompt = _read_line("prompt> ").strip()
                if not prompt:
                    print("empty prompt")
                    continue
                max_tokens = _input_int("max_tokens", 128)
                temperature = _input_float("temperature", 0.0)
                run_chat_once(cfg, prompt, max_tokens=max_tokens, temperature=temperature, as_json=False)
            elif choice == "8":
                _run_chat_session(cfg)
            elif choice == "9":
                path = Path(_input_default("profile_path", str(DEFAULT_PROFILE)))
                save_profile(cfg, path)
                print("saved:", path)
            elif choice == "10":
                path = Path(_input_default("profile_path", str(DEFAULT_PROFILE)))
                if not path.exists():
                    print("file not found:", path)
                    continue
                loaded = load_profile(path)
                if loaded.provider not in ADAPTERS:
                    print("invalid provider in profile:", loaded.provider)
                    continue
                cfg = loaded
                print("loaded:", path)
            else:
                print("unknown menu")
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone TUI app for LLM API tests.")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to .env file (default: app/llm_tui/.env).",
    )
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "openai_compatible"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:18010"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemma-4-26B-A4B-it"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("LLM_TIMEOUT_SEC", "120")))
    parser.add_argument("--once", default="", help="Run one prompt and exit (non-interactive).")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Use with --once for JSON output.")
    parser.add_argument("--chat", action="store_true", help="Start directly in chat session mode.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output.")
    # Parse once to get env file, then load env, then parse final args
    pre_args, _ = parser.parse_known_args()
    loaded_env = load_env_with_fallback(Path(pre_args.env_file), DEFAULT_ENV_EXAMPLE_FILE)
    parser.set_defaults(
        provider=os.environ.get("LLM_PROVIDER", "openai_compatible"),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:18010"),
        model=os.environ.get("LLM_MODEL", "gemma-4-26B-A4B-it"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        timeout_sec=float(os.environ.get("LLM_TIMEOUT_SEC", "120")),
    )
    args = parser.parse_args()

    if args.provider not in ADAPTERS:
        print(f"unsupported provider: {args.provider}", file=sys.stderr)
        return 2

    cfg = AppConfig(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    if loaded_env:
        print(f"config source: {loaded_env}")
    if args.chat:
        try:
            _run_chat_session(cfg, stream=args.stream)
            return 0
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            return 0
    if args.once:
        return run_chat_once(
            cfg,
            prompt=args.once,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            as_json=args.json,
            stream=args.stream,
        )
    return tui_loop(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
