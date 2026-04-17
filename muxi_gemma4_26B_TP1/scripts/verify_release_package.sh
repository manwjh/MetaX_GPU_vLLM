#!/usr/bin/env bash
# 在 **发布包根目录**（含 `scripts/`、`Dockerfile` 的目录）执行静态检查：
# - 所有 `scripts/*.sh` 的 bash -n
# - 所有 `scripts/*.py` 的 python3 -m py_compile
#
# 用法：
#   cd /path/to/muxi_gemma4_26B_TP1
#   bash scripts/verify_release_package.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "== verify_release_package root=${ROOT} =="

fail=0
while IFS= read -r -d '' s; do
  if ! bash -n "$s"; then
    fail=1
  fi
done < <(find "${ROOT}/scripts" -maxdepth 1 -name '*.sh' -print0 2>/dev/null)

while IFS= read -r -d '' p; do
  if ! python3 -m py_compile "$p"; then
    fail=1
  fi
done < <(find "${ROOT}/scripts" -maxdepth 1 -name '*.py' -print0 2>/dev/null)

if [[ "${fail}" -ne 0 ]]; then
  echo "verify_release_package: FAIL" >&2
  exit 1
fi

echo "verify_release_package: OK"
exit 0
