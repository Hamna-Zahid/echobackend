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
        # REDIS_TOKEN is only used when the password is NOT embedded in the URL.
        # Upstash typically embeds credentials in the URL (rediss://user:pass@host:port),
        # so we only pass the explicit token when the URL has no auth component.
        redis_token = os.environ.get("REDIS_TOKEN", "")

        if redis_url.startswith("https://") or redis_url.startswith("rediss://"):
            # Check whether password is already embedded (user:pass@host)
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            has_embedded_auth = bool(parsed.password)
            kwargs: dict = {"decode_responses": True, "ssl_cert_reqs": None}
            if not has_embedded_auth and redis_token:
                kwargs["password"] = redis_token
            _redis = aioredis.from_url(redis_url, **kwargs)
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
