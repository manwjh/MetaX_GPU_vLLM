# 测试报告模板（复制为 `YYYY-MM-DD_<标签>.md`）

## 元数据

| 项 | 值 |
|----|-----|
| 日期 | YYYY-MM-DD |
| 发布包版本 / 镜像 tag | 例：`muxi_gemma4_26B_TP1:2026-04-17` |
| 基镜像 `BASE_IMAGE` | 沐曦 `vllm-metax` 全名 |
| 执行人 / 环境 | 本机 / 长沙 GPU |

## 1. 静态检查（无 GPU）

| 检查项 | 命令 / 说明 | 结果 |
|--------|-------------|------|
| 包内脚本 `bash -n` | `scripts/verify_release_package.sh` | PASS / FAIL |
| 包内 Python `py_compile` | 同上 | PASS / FAIL |
| Monorepo `make check` | 仓库根（仅当在完整仓库中开发时） | PASS / SKIP |

## 2. 镜像构建

| 检查项 | 结果 |
|--------|------|
| `docker build` 成功 | YES / NO |
| 构建日志中 `transformers_ok 5.5.0`（或等价） | YES / NO |
| pip 对 `vllm … transformers<5` 的告警（预期） | 已记录 / N/A |

## 3. 配置与权重（不启动完整引擎）

| 检查项 | 结果 |
|--------|------|
| `AutoConfig.from_pretrained` + `model_type=gemma4` | PASS / FAIL |
| 权重宿主机路径（示例） |  |

## 4. 可选：运行时（需 GPU + 权重）

| 检查项 | 结果 |
|--------|------|
| 容器 `GET /v1/models` 200 |  |
| `verify_service.sh` 或 curl chat |  |
| 可选：monorepo `changsha_autoverify.sh`（长沙） |  |

## 5. 结论

- **可发布：** YES / NO（含附加条件）
- **已知问题：** 

## 6. 证据路径

- 日志 / 截图 / 导出文件路径（可填仓库内 `docs/muxi/exports/...`）
