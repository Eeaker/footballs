#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python; fi
"$PY" scripts/system_check.py || true
echo
echo "--- 产品/API验收 ---"
"$PY" scripts/verify_product.py
