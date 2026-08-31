from __future__ import annotations

import atexit
import os
import socket
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.config import SYSTEM_VERSION

RUNTIME = ROOT / "runtime"
PID_FILE = RUNTIME / "server.pid"
URL_FILE = RUNTIME / "server.url"


def existing_server_url() -> str | None:
    """Reuse the recorded healthy instance instead of silently starting another port."""
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        url = URL_FILE.read_text(encoding="utf-8").strip().rstrip("/")
        os.kill(pid, 0)
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.5) as response:
            payload = response.read(512)
        if b'"ok":true' in payload and SYSTEM_VERSION.encode("utf-8") in payload:
            return url
    except (OSError, ValueError):
        return None
    return None


def free_port(preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("8000-8019 端口均被占用")


def cleanup_runtime_marker() -> None:
    for path in (PID_FILE, URL_FILE):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    RUNTIME.mkdir(parents=True, exist_ok=True)
    running_url = existing_server_url()
    if running_url:
        print(f"系统已经运行：{running_url}")
        webbrowser.open(running_url)
        raise SystemExit(0)
    port = free_port(int(os.getenv("FOOTBALL_INSIGHT_PORT", "8000")))
    url = f"http://127.0.0.1:{port}"
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    URL_FILE.write_text(url, encoding="utf-8")
    atexit.register(cleanup_runtime_marker)
    print("=" * 64)
    print(f"Football Insight V{SYSTEM_VERSION} Windows 正式系统")
    print(f"系统地址: {url}")
    print("关闭本窗口即可停止服务。")
    print("演示时可使用 PRESENT_WINDOWS.vbs 静默启动。")
    print("=" * 64)

    if sys.platform == "win32":
        import asyncio
        original_handler = asyncio.get_event_loop().call_exception_handler
        def _silence_connection_reset(loop, context):
            exc = context.get("exception")
            if isinstance(exc, ConnectionResetError):
                return
            original_handler(loop, context) if original_handler else None
        asyncio.get_event_loop().call_exception_handler(_silence_connection_reset)

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run("app.app:app", host="127.0.0.1", port=port, log_level="warning")
    finally:
        cleanup_runtime_marker()
