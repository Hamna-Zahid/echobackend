"""
Long-term learner memory — the retrieval layer that makes Echo feel like the
same coach every time, instead of a stranger asking the same opening question.

WHERE IT LIVES
    public.users.memory  (jsonb, see db/migrations/add_user_memory.sql)
    One row per learner, rewritten after each session. No vector DB is needed:
    a learner's history is small and highly structured, so the whole memory is
    a few hundred tokens and is injected wholesale. That is the correct
    retrieval strategy at this size — embeddings only pay off once the corpus
    is too big to fit in the prompt.

HOW IT IS USED (the RAG loop)
    session start -> load_memory()  -> build_memory_context() -> system prompt
    session end   -> extract_session_memory() (LLM) -> merge -> save_memory()

Everything is best-effort: any failure degrades to "no memory" and the session
still runs exactly as before.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from db.supabase import get_supabase
from services.llm import get_groq_client, FAST_MODEL

logger = logging.getLogger("echo.memory")

MAX_LIST = 6  # keep each remembered list short so the prompt stays tight


# ── Read ──────────────────────────────────────────────────────────────────────

def load_memory(user_id: str) -> dict:
    try:
        res = get_supabase().table("users").select("memory").eq("id", user_id).single().execute()
        mem = (res.data or {}).get("memory")
        return mem if isinstance(mem, dict) else {}
    except Exception as e:
        logger.debug("memory load failed: %s", e)
        return {}


def build_memory_context(memory: dict) -> str:
    """Render stored memory into the prompt block the coach is grounded in."""
    if not memory:
        return ""

    lines = []
    n = memory.get("session_count") or 0
    if n:
        mins = memory.get("total_minutes") or 0
        lines.append(f"You have coached this learner {n} time(s) before (~{mins} minutes total).")

    last = (memory.get("last_session_summary") or "").strip()
    if last:
        lines.append(f"LAST SESSION: {last}")

    goals = memory.get("goals") or []
    if goals:
        lines.append(f"WHAT THEY WANT FROM ENGLISH: {', '.join(goals[:MAX_LIST])}")

    topics = memory.get("preferred_topics") or []
    if topics:
        lines.append(f"TOPICS THEY ENJOY: {', '.join(topics[:MAX_LIST])}")

    facts = memory.get("personal_facts") or []
    if facts:
        lines.append(f"WHAT YOU KNOW ABOUT THEM: {'; '.join(facts[:MAX_LIST])}")

    struggles = memory.get("persistent_struggles") or []
    if struggles:
        lines.append(f"RECURRING LANGUAGE STRUGGLES: {', '.join(struggles[:MAX_LIST])}")

    mastered = memory.get("mastered") or []
    if mastered:
        lines.append(f"ALREADY IMPROVED (don't re-teach): {', '.join(mastered[:MAX_LIST])}")

    notes = (memory.get("personality_notes") or "").strip()
    if notes:
        lines.append(f"HOW THEY SPEAK: {notes}")

    open_thread = (memory.get("open_thread") or "").strip()

    block = "━━━ WHAT YOU REMEMBER ABOUT THIS LEARNER ━━━\n" + "\n".join(lines)
    block += (
        "\n\nUSE THIS MEMORY: open by picking the conversation back up like a coach who "
        "knows them — refer naturally to something specific from last time, then ask what "
        "they'd like to work on or talk about TODAY. Never re-introduce yourself as if you "
        "were meeting for the first time, and never repeat a generic scripted opener."
    )
    if open_thread:
        block += f"\n\nUNFINISHED THREAD TO PICK UP: {open_thread}"
    return block


# ── Write ─────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """You maintain a language coach's private notes about ONE learner.

Given the previous notes and a transcript of the session that just finished, return the UPDATED notes as JSON only:

{
  "last_session_summary": "<2 sentences: what they talked about and how they did>",
  "open_thread": "<something specific left unfinished or worth following up next time, or ''>",
  "goals": ["<why they're learning English>"],
  "preferred_topics": ["<topics they engaged with>"],
  "personal_facts": ["<durable facts: job, city, family, studies, plans>"],
  "persistent_struggles": ["<language issues seen again this session>"],
  "mastered": ["<issues that seem fixed now>"],
  "personality_notes": "<1 sentence on how they speak and what motivates them>"
}

Rules: MERGE with the previous notes, don't discard still-true facts. Keep every list to at most 6 short items. Only record what the transcript actually supports — never invent. JSON only."""


async def extract_session_memory(previous: dict, transcript: str) -> Optional[dict]:
    """Ask the fast model to fold this session into the learner's notes."""
    if not transcript.strip():
        return None
    try:
        client = get_groq_client()
        res = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content":
                    f"PREVIOUS NOTES:\n{json.dumps(previous or {}, ensure_ascii=False)}\n\n"
                    f"SESSION TRANSCRIPT:\n{transcript[:6000]}"},
            ],
            stream=False,
            max_tokens=700,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        out = json.loads(res.choices[0].message.content)
        return out if isinstance(out, dict) else None
    except Exception as e:
        logger.debug("memory extraction failed: %s", e)
        return None


def _merge(previous: dict, fresh: dict, duration_seconds: int) -> dict:
    """Merge extracted notes over the old ones and bump the counters."""
    merged = dict(previous or {})
    for key in ("last_session_summary", "open_thread", "personality_notes"):
        if fresh.get(key):
            merged[key] = fresh[key]
    for key in ("goals", "preferred_topics", "personal_facts",
                "persistent_struggles", "mastered"):
        vals = fresh.get(key)
        if isinstance(vals, list) and vals:
            seen, out = set(), []
            for v in vals:
                v = str(v).strip()
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    out.append(v)
            merged[key] = out[:MAX_LIST]

    # Anything now mastered should stop being listed as a struggle.
    mastered = {m.lower() for m in merged.get("mastered", [])}
    if mastered:
        merged["persistent_struggles"] = [
            s for s in merged.get("persistent_struggles", []) if s.lower() not in mastered
        ]

    merged["session_count"] = int(merged.get("session_count") or 0) + 1
    merged["total_minutes"] = int(merged.get("total_minutes") or 0) + max(1, duration_seconds // 60)
    merged["last_session_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def save_memory(user_id: str, memory: dict) -> None:
    try:
        get_supabase().table("users").update({"memory": memory}).eq("id", user_id).execute()
    except Exception as e:
        logger.debug("memory save failed: %s", e)


async def update_after_session(
    user_id: str, previous: dict, transcript: str, duration_seconds: int
) -> None:
    """Full write path: extract -> merge -> persist. Never raises."""
    try:
        fresh = await extract_session_memory(previous, transcript)
        if not fresh:
            return
        save_memory(user_id, _merge(previous, fresh, duration_seconds))
    except Exception as e:
        logger.debug("memory update failed: %s", e)
