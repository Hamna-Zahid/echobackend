"""
Entitlements & subscription — free vs premium gating.

Tiers (stored in users.memory["subscription"] = {tier, until, since}):
  FREE     — diagnostic + the leveled path + Training & Conversation modes
             + 1 coaching session/day + basic correction cards + streaks/levels.
  PREMIUM  — unlimited sessions, ALL modes, full roadmap depth, detailed
             post-session reports, accent coaching, progress analytics.

Premium auto-expires when `until` passes (treated as free thereafter).
"""

from datetime import datetime, timezone

from db.supabase import get_supabase
from services.user_state import get_memory, update_memory

FREE_MODES = {"diagnostic", "training", "conversation"}
ALL_MODES = {"diagnostic", "training", "conversation", "interview", "professional", "accent"}
FREE_DAILY_COACHING_SESSIONS = 1   # diagnostic is always allowed and excluded from this count


# ── Tier resolution ───────────────────────────────────────────────────────────

def get_subscription(memory: dict) -> dict:
    sub = memory.get("subscription") if isinstance(memory, dict) else None
    if not isinstance(sub, dict):
        return {"tier": "free", "until": None}
    tier = sub.get("tier", "free")
    until = sub.get("until")
    if tier == "premium" and until:
        try:
            if datetime.fromisoformat(until.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                return {"tier": "free", "until": until, "expired": True}
        except Exception:
            pass
    return {"tier": tier, "until": until}


def is_premium(memory: dict) -> bool:
    return get_subscription(memory).get("tier") == "premium"


def entitlements(memory: dict) -> dict:
    premium = is_premium(memory)
    return {
        "tier": "premium" if premium else "free",
        "is_premium": premium,
        "allowed_modes": sorted(ALL_MODES if premium else FREE_MODES),
        "daily_coaching_limit": None if premium else FREE_DAILY_COACHING_SESSIONS,
        "features": {
            "all_modes": premium,
            "unlimited_sessions": premium,
            "detailed_reports": premium,
            "accent_coaching": premium,
            "progress_analytics": premium,
            "full_roadmap": premium,
        },
    }


# ── Daily usage ───────────────────────────────────────────────────────────────

def _coaching_sessions_today(user_id: str) -> int:
    """Count non-diagnostic sessions the user started today (UTC)."""
    try:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        res = (
            get_supabase()
            .table("sessions")
            .select("id, mode, started_at", count="exact")
            .eq("user_id", user_id)
            .neq("mode", "diagnostic")
            .gte("started_at", start.isoformat())
            .execute()
        )
        return res.count or 0
    except Exception:
        return 0


def can_start_session(user_id: str, mode: str, memory: dict | None = None) -> dict:
    """
    Decide whether the user may start a session in `mode`.
    Returns {allowed: bool, reason: str|None, upgrade: bool}.
    """
    mem = memory if isinstance(memory, dict) else get_memory(user_id)
    if is_premium(mem):
        return {"allowed": True, "reason": None, "upgrade": False}

    if mode not in FREE_MODES:
        return {"allowed": False, "reason": "premium_mode", "upgrade": True}

    if mode == "diagnostic":
        return {"allowed": True, "reason": None, "upgrade": False}

    used = _coaching_sessions_today(user_id)
    if used >= FREE_DAILY_COACHING_SESSIONS:
        return {"allowed": False, "reason": "daily_limit", "upgrade": True}

    return {"allowed": True, "reason": None, "upgrade": False}


# ── Mock subscription management (real IAP wired at store-submission time) ─────

def set_premium(user_id: str, months: int = 1) -> dict:
    """Flip the user to premium (mock purchase). Returns the new subscription."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    until = now + timedelta(days=30 * max(1, months))
    sub = {"tier": "premium", "since": now.isoformat(), "until": until.isoformat(), "mock": True}

    def mutate(mem: dict):
        mem["subscription"] = sub
    update_memory(user_id, mutate)
    return sub


def cancel_premium(user_id: str) -> dict:
    sub = {"tier": "free", "until": None}

    def mutate(mem: dict):
        mem["subscription"] = sub
    update_memory(user_id, mutate)
    return sub
