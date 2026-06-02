"""
User memory service — persistent AI memory per learner.

Echo remembers each user's learning journey across sessions:
  - Which grammar patterns they've mastered
  - What they keep struggling with
  - Topics they enjoy talking about
  - How they respond to feedback (personality notes)
  - A short summary of their last session

Memory is stored as a JSONB column on the users table (no extra table needed).
It's loaded at session start and injected into the system prompt.
It's updated at session end with a fast LLM summarisation call.

Memory shape:
  {
    "session_count":        int,
    "total_minutes":        int,
    "mastered":             list[str],
    "persistent_struggles": list[str],
    "preferred_topics":     list[str],
    "personality_notes":    str,
    "last_session_summary": str,
    "last_session_mode":    str,
    "last_session_at":      str (ISO-8601),
  }
"""

import json
from datetime import datetime, timezone
from typing import Optional

from db.supabase import get_supabase


# ── Load ──────────────────────────────────────────────────────────────────────

def load_user_memory(user_id: str) -> dict:
    """
    Load the user's memory from Supabase.
    Returns an empty dict if no memory exists yet (first session).
    Never raises — any DB error returns empty memory gracefully.
    """
    try:
        supabase = get_supabase()
        res = (
            supabase.table("users")
            .select("memory")
            .eq("id", user_id)
            .single()
            .execute()
        )
        mem = (res.data or {}).get("memory") or {}
        return mem if isinstance(mem, dict) else {}
    except Exception:
        return {}


# ── Build prompt context ──────────────────────────────────────────────────────

def build_memory_context(memory: dict, user_name: str = "") -> str:
    """
    Convert raw memory dict into a compact prompt section.
    Injected at the top of every system prompt so the AI knows the learner.
    Returns empty string if memory is empty (first session).
    """
    if not memory:
        return ""

    parts = []
    name_str = f" ({user_name.strip()})" if user_name else ""

    session_count = memory.get("session_count", 0)
    total_minutes = memory.get("total_minutes", 0)
    if session_count:
        parts.append(f"This learner{name_str} has completed {session_count} session(s), ~{total_minutes} minutes total.")

    mastered = memory.get("mastered", [])
    if mastered:
        parts.append(f"Mastered: {', '.join(mastered)}.")

    struggles = memory.get("persistent_struggles", [])
    if struggles:
        parts.append(f"Still struggles with: {', '.join(struggles)}. Focus correction energy here.")

    topics = memory.get("preferred_topics", [])
    if topics:
        parts.append(f"Enjoys talking about: {', '.join(topics)}. Use these topics naturally.")

    notes = memory.get("personality_notes", "")
    if notes:
        parts.append(f"Coaching notes: {notes}")

    last_summary = memory.get("last_session_summary", "")
    if last_summary:
        parts.append(f"Last session: {last_summary}")

    if not parts:
        return ""

    return (
        "━━━ LEARNER MEMORY (private — do not quote back to user) ━━━\n"
        + "\n".join(parts)
        + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )


# ── Save after session ────────────────────────────────────────────────────────

async def update_user_memory(
    user_id: str,
    existing_memory: dict,
    session_transcript: str,
    session_mode: str,
    session_minutes: int,
    top_errors: list[dict],
) -> None:
    """
    Update user memory after a session ends.
    Uses Groq Llama to extract insights from the transcript, then merges
    with existing memory and saves back to Supabase.
    Never raises — memory update failures are silent.
    """
    if not session_transcript or len(session_transcript.strip()) < 100:
        return  # Too short to learn from

    try:
        insights = await _extract_session_insights(session_transcript, top_errors)
        if not insights:
            return

        # Merge insights into the FRESHEST memory (re-read), so concurrent
        # end-of-session writes (e.g. XP/gamification) are never clobbered.
        from services.user_state import update_memory

        def _mutate(mem: dict):
            merged = _merge_memory(mem, insights, session_mode, session_minutes)
            mem.clear()
            mem.update(merged)

        update_memory(user_id, _mutate)
    except Exception:
        pass  # Memory update is best-effort — never block session close


async def _extract_session_insights(transcript: str, top_errors: list[dict]) -> Optional[dict]:
    """Delegate to the central LLM service (Gemini Flash primary, Groq Llama fallback)."""
    from services.llm import extract_session_insights
    return await extract_session_insights(transcript, top_errors)


def _merge_memory(
    existing: dict,
    insights: dict,
    session_mode: str,
    session_minutes: int,
) -> dict:
    """
    Merge new session insights into existing memory.
    Lists are merged (deduped, capped at 8 items). Strings are replaced.
    """
    def merge_list(old: list, new: list, cap: int = 8) -> list:
        combined = list(dict.fromkeys(old + new))  # preserve order, dedupe
        return combined[:cap]

    updated = dict(existing)

    updated["session_count"] = existing.get("session_count", 0) + 1
    updated["total_minutes"] = existing.get("total_minutes", 0) + session_minutes

    if insights.get("mastered"):
        updated["mastered"] = merge_list(
            existing.get("mastered", []), insights["mastered"]
        )
        # Remove from struggles if now mastered
        struggles = existing.get("persistent_struggles", [])
        mastered_lower = {m.lower() for m in updated["mastered"]}
        updated["persistent_struggles"] = [
            s for s in struggles if s.lower() not in mastered_lower
        ]

    if insights.get("persistent_struggles"):
        updated["persistent_struggles"] = merge_list(
            updated.get("persistent_struggles", []), insights["persistent_struggles"]
        )

    if insights.get("preferred_topics"):
        updated["preferred_topics"] = merge_list(
            existing.get("preferred_topics", []), insights["preferred_topics"]
        )

    if insights.get("personality_notes"):
        updated["personality_notes"] = insights["personality_notes"]

    if insights.get("last_session_summary"):
        updated["last_session_summary"] = insights["last_session_summary"]

    updated["last_session_mode"] = session_mode
    updated["last_session_at"] = datetime.now(timezone.utc).isoformat()

    return updated


def _save_memory(user_id: str, memory: dict) -> None:
    """Persist updated memory to Supabase users table."""
    try:
        supabase = get_supabase()
        supabase.table("users").update({"memory": memory}).eq("id", user_id).execute()
    except Exception:
        pass
