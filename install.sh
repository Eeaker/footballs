#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
else
  PY=""
  for candidate in python3.11 python3.10 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
fi
if [ -z "$PY" ]; then echo "未找到 Python，请安装 Python 3.10–3.12"; exit 1; fi
"$PY" -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -r requirements.txt
chmod +x start.sh check.sh
echo
echo "[Football Insight] 安装完成"
echo "Python: $($PY --version 2>&1)"
echo "1) 运行 ./check.sh 检查环境"
echo "2) 运行 ./start.sh 启动系统"
echo "3) 可在系统状态页上传 yolov8x.pt"
