# `evidence/`

| 文件 | 说明 |
|------|------|
| `00_monorepo_make_check.log` | 打 tarball 前在 **changsha_gpu 仓库根**执行 `make check` 的输出（含补丁锚点校验） |
| `01_package_static_verify.log` | 对**本包解压目录**执行 `bash scripts/verify_release_package.sh` 的输出 |
| `02_changsha_field_excerpt.txt` | 长沙现场发版轮次 `changsha_autoverify` 关键行摘录（见文件头说明） |

说明：`00` 可能含本机构建机路径，不影响结论；以 **`verify_release_package: OK`** 与矩阵 **`MATRIX_ACCEPT_STATUS PASS`** 行为准。
