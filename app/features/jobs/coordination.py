from __future__ import annotations

import os
import secrets
from dataclasses import dataclass


@dataclass
class JobLease:
    key: str
    token: str
    client: object | None = None

    def release(self) -> None:
        if self.client is None:
            return
        # Delete only our lease; never release another worker's lock.
        self.client.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
            1, self.key, self.token,
        )


def acquire_job_lease(project_id: str, ttl_seconds: int = 24 * 60 * 60) -> JobLease | None:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return JobLease(f"local:{project_id}", secrets.token_hex(16), None)
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("缺少 Redis 驱动，请使用一键容器部署") from exc
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
    key, token = f"football-insight:job:{project_id}", secrets.token_hex(16)
    if not client.set(key, token, nx=True, ex=ttl_seconds):
        return None
    return JobLease(key, token, client)

