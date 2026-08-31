#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python; fi
echo "[Football Insight] V2.2 http://127.0.0.1:${PORT}"
"$PY" -m uvicorn app.app:app --host 0.0.0.0 --port "$PORT"
