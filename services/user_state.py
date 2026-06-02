"""
User state store — atomic read-merge-write helpers over users.memory (jsonb).

We keep persistent app state (learning memory, gamification, subscription) inside
the existing users.memory jsonb column so no schema migration is required.

Namespacing inside memory:
  memory["gamification"] = {xp, level, streak_days, last_practice_date, daily_xp,
                            daily_goal_xp, daily_date, longest_streak, total_sessions}
  memory["subscription"] = {tier: "free"|"premium", until: ISO|None, since: ISO}
  (learning insights stay at top level: mastered, persistent_struggles, etc.)

All writes go through update_memory(), which RE-READS the current row before merging
so concurrent end-of-session writes (insights vs XP) don't clobber each other.
"""

from typing import Callable

from db.supabase import get_supabase


def get_memory(user_id: str) -> dict:
    """Read the full memory dict for a user. Returns {} on any error."""
    try:
        res = (
            get_supabase()
            .table("users")
            .select("memory")
            .eq("id", user_id)
            .single()
            .execute()
        )
        mem = (res.data or {}).get("memory") or {}
        return mem if isinstance(mem, dict) else {}
    except Exception:
        return {}


def save_memory(user_id: str, memory: dict) -> None:
    """Persist the full memory dict. Best-effort."""
    try:
        get_supabase().table("users").update({"memory": memory}).eq("id", user_id).execute()
    except Exception:
        pass


def update_memory(user_id: str, mutate: Callable[[dict], None]) -> dict:
    """
    Atomic-ish read-modify-write: re-read the freshest memory, apply `mutate`
    (which edits the dict in place), then write it back. This avoids the
    stale-state clobber where two end-of-session writers each overwrite memory.
    """
    mem = get_memory(user_id)
    try:
        mutate(mem)
    except Exception:
        return mem
    save_memory(user_id, mem)
    return mem
