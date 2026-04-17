# `evidence/`

| 文件 | 说明 |
|------|------|
| `00_monorepo_make_check.log` | 打 tarball 前在 **changsha_gpu 仓库根**执行 `make check` 的输出（含补丁锚点校验） |
| `01_package_static_verify.log` | 对**本包解压目录**执行 `bash scripts/verify_release_package.sh` 的输出 |
| `02_changsha_field_excerpt.txt` | 长沙现场发版轮次 `changsha_autoverify` 关键行摘录（见文件头说明） |
| `03_server_runtime_snapshot_2026-04-17.log` | 服务器运行态快照：`docker top` 参数、`ENGINE_ENV_PROBE` 环境变量（含 `VLLM_GEMMA4_METAX_CHAT_PROMPT_ALIGN=0`） |
| `04_app_probe_nihao_2026-04-17.txt` | 使用包内 `app/llm_tui` 对“你好呀”的在线探针输出 |
| `05_app_probe_math_stream_2026-04-17.txt` | 使用包内 `app/llm_tui --stream` 对“1加1等于几？”的在线探针输出 |

说明：`00` 可能含本机构建机路径，不影响结论；以 **`verify_release_package: OK`** 与矩阵 **`MATRIX_ACCEPT_STATUS PASS`** 行为准。
