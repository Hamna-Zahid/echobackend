import json
import os
from typing import Optional

import redis.asyncio as aioredis

_redis: aioredis.Redis | None = None

SESSION_TTL = 7200  # 2 hours


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_token = os.environ.get("REDIS_TOKEN", "")

        # Upstash Redis uses HTTPS URL with token auth
        if redis_url.startswith("https://") or redis_url.startswith("rediss://"):
            _redis = aioredis.from_url(
                redis_url,
                password=redis_token,
                decode_responses=True,
                ssl_cert_reqs=None,
            )
        else:
            _redis = aioredis.from_url(redis_url, decode_responses=True)
    return _redis


def _key(session_id: str) -> str:
    return f"session:{session_id}"


async def get_session_state(session_id: str) -> Optional[dict]:
    r = get_redis()
    data = await r.get(_key(session_id))
    if data is None:
        return None
    return json.loads(data)


async def set_session_state(session_id: str, state: dict) -> None:
    r = get_redis()
    await r.set(_key(session_id), json.dumps(state), ex=SESSION_TTL)


async def delete_session_state(session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(session_id))


async def update_session_field(session_id: str, field: str, value) -> None:
    """Atomic field update — read-modify-write with optimistic approach."""
    state = await get_session_state(session_id)
    if state is not None:
        state[field] = value
        await set_session_state(session_id, state)
