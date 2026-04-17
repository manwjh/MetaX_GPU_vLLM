# LLM TUI App

终端交互测试工具，当前支持 OpenAI-compatible 接口（可直接连 vLLM/OpenAI 风格服务）。

## 快速开始

```bash
# 可选：复制一份本地配置
cp app/llm_tui/.env.example app/llm_tui/.env

# 启动菜单模式
python3 app/llm_tui/main.py
```

配置加载顺序：

1. `--env-file <path>`（若传入）
2. 默认 `app/llm_tui/.env`
3. 若 `.env` 不存在，自动回退 `app/llm_tui/.env.example`

## 常用命令

菜单模式：

```bash
python3 app/llm_tui/main.py
```

直接进入持续聊天：

```bash
python3 app/llm_tui/main.py --chat
```

持续聊天 + 流式输出：

```bash
python3 app/llm_tui/main.py --chat --stream
```

一次性调用（非交互）：

```bash
python3 app/llm_tui/main.py \
  --provider openai_compatible \
  --base-url http://127.0.0.1:18010 \
  --model gemma-4-26B-A4B-it \
  --once "用一句话介绍长沙。"
```

一次性调用 + 流式输出：

```bash
python3 app/llm_tui/main.py --once "hello" --stream
```

JSON 输出（非交互）：

```bash
python3 app/llm_tui/main.py --once "hello" --json
```

## 菜单交互说明

- `5) Set api_key`：输入规则为 `Enter=保持`、`-=清空`、其他=新值
- `8) Chat session mode`：输入 `/exit` 退出会话
- `Ctrl+C` / `Ctrl+D`：安全退出
- 聊天模式启动前会先做一次 `/v1/models` 健康检查并打印目标配置

## 环境变量

- `LLM_PROVIDER`（默认 `openai_compatible`）
- `LLM_BASE_URL`（如 `http://127.0.0.1:18010`）
- `LLM_MODEL`
- `LLM_API_KEY`（可空）
- `LLM_TIMEOUT_SEC`

## 设计与扩展

- 通过 `ServiceAdapter` 抽象 provider
- 当前实现：`OpenAICompatibleAdapter`
- 后续接入其他服务：新增 adapter 并注册到 `ADAPTERS`

## 注意事项

- 流式模式下，部分 OpenAI-compatible 服务不会返回完整 `usage`，看到 `usage: {}` 属正常行为。
- 当前默认是“忠实显示服务端输出”，不在客户端做流式内容截断或去重修饰。
- 输入层优先使用 `prompt_toolkit`（若环境可用）以改善中文输入法与回退删除体验；不可用时自动回退 Python 内置 `input()`。
