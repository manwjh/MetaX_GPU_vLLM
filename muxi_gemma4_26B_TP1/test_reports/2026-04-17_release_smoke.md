# 测试报告：`muxi_gemma4_26B_TP1` 发版前一轮（2026-04-17）

## 元数据

| 项 | 值 |
|----|-----|
| 日期 | 2026-04-17 |
| 发布包 | `dist/muxi_gemma4_26B_TP1` / `muxi_gemma4_26B_TP1.tar.gz`（由 `scripts/muxi_gemma4/build_muxi_gemma4_26B_TP1_release.sh` 生成） |
| 基镜像 | `cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.15.0-maca.ai3.5.3.203-torch2.8-py310-ubuntu22.04-amd64` |
| 长沙现场 | `wjh-vllm-gemma4`，`122.207.82.8` |

## 1. 静态检查（无 GPU）

| 检查项 | 结果 |
|--------|------|
| 发布包内 `bash scripts/verify_release_package.sh` | **PASS**（`verify_release_package: OK`） |
| Monorepo `make check`（仓库根） | **PASS**（`gemma4-check` + `verify_metax_gemma4_patch_anchors: ok`） |

## 2. 镜像与 Transformers（历史 + 本轮）

| 检查项 | 结果 |
|--------|------|
| Dockerfile 内 **`pip install transformers==5.5.0`** | 已在 `Dockerfile` 固化；长沙曾 **`docker build`** 成功，基镜像 **4.57.6 → 5.5.0** |
| pip 对 `vllm … transformers<5` 告警 | **预期**（Gemma4 需 5.x）；见包内 `README.md` §10 |

## 3. 长沙在线服务（与发布包脚本同源）

**说明：** 线容器未必由本 tar 直接 build，但栈一致（vLLM 0.15 + 补丁思路 + `transformers 5.5.0`）。

| 检查项 | 结果 |
|--------|------|
| `MATRIX_QUICK=1 bash scripts/muxi_gemma4/changsha_autoverify.sh` | **PASS**（`MATRIX_ACCEPT_STATUS PASS`） |
| 证据日志 | `docs/muxi/exports/changsha_autoverify_release_round_2026-04-17.log` |

## 4. 未在本轮重复执行的项

| 项 | 说明 |
|----|------|
| 独占 GPU 的 `docker build` + 全量起服 | 已在 **`Gemma4-实验笔记.md` 续三十二** 记录；需要时可对 **`muxi_gemma4_tp1_verify:local`** 重复 |
| `MATRIX_STRICT=1` | 发版前建议在 monorepo 执行一次 |

## 5. 结论

- **可发布：** **YES**（在「必须 `docker build` 安装 Transformers 5.5.0」前提下）。
- **随包交付：** `test_reports/` 目录、`scripts/verify_release_package.sh`。
