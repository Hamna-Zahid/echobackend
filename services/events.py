"""
Lightweight event logging for the admin panel's observability view.

Every important backend event is written to public.app_events: session lifecycle
plus the STT / LLM / feedback stages with latency and ok|error status. Writes are
best-effort and fire-and-forget — logging must never slow down or break a session.
"""

import asyncio
import logging
from typing import Optional

from db.supabase import get_supabase

logger = logging.getLogger("echo.events")


def _insert(row: dict) -> None:
    try:
        get_supabase().table("app_events").insert(row).execute()
    except Exception as e:
        logger.debug("event log failed: %s", e)


def log_event(
    event_type: str,
    status: str = "ok",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    latency_ms: Optional[int] = None,
    detail: Optional[dict] = None,
) -> None:
    """
    Fire-and-forget event log. Schedules the DB insert on a thread so it never
    blocks the event loop or the live session.
    """
    row = {
        "event_type": event_type,
        "status": status,
        "user_id": user_id,
        "session_id": session_id,
        "latency_ms": latency_ms,
        "detail": detail,
    }
    try:
        asyncio.create_task(asyncio.to_thread(_insert, row))
    except RuntimeError:
        # No running loop (e.g. called from sync context) — insert inline.
        _insert(row)
