from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from datetime import date, datetime, timezone, timedelta

from db.supabase import get_supabase, get_current_user

router = APIRouter()


def _compute_streak(sessions: list) -> int:
    """
    Count the number of consecutive days (ending today or yesterday) that
    had at least one session. Returns 0 if no streak exists.
    """
    if not sessions:
        return 0

    # Build a set of unique session dates
    session_dates: set[date] = set()
    for s in sessions:
        raw = s.get("started_at")
        if raw:
            try:
                session_dates.add(date.fromisoformat(raw[:10]))
            except ValueError:
                pass

    if not session_dates:
        return 0

    today = datetime.now(timezone.utc).date()

    # Streak may start from today or from yesterday (if user hasn't practiced yet today)
    streak = 0
    check = today
    while check in session_dates:
        streak += 1
        check -= timedelta(days=1)

    # If no session today, try starting from yesterday
    if streak == 0:
        check = today - timedelta(days=1)
        while check in session_dates:
            streak += 1
            check -= timedelta(days=1)

    return streak


@router.get("/")
async def list_sessions(user=Depends(get_current_user)):
    supabase = get_supabase()
    res = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user["id"])
        .order("started_at", desc=True)
        .limit(20)
        .execute()
    )
    return {"sessions": res.data}


# NOTE: Static sub-paths (/errors/..., /progress/...) MUST come before
# the wildcard /{session_id} route so FastAPI's router matches them first.

@router.get("/errors/recurring")
async def get_recurring_errors(user=Depends(get_current_user)):
    supabase = get_supabase()
    res = (
        supabase.table("errors")
        .select("*")
        .eq("user_id", user["id"])
        .eq("resolved", False)
        .order("recurrence_count", desc=True)
        .limit(10)
        .execute()
    )
    return {"errors": res.data}


@router.get("/progress/summary")
async def get_progress(user=Depends(get_current_user)):
    supabase = get_supabase()

    # Fetch up to 90 sessions for streak calculation + recent display.
    # Falls back to the original columns if the add_admin.sql migration
    # (overall_score/wpm/filler_count) hasn't been applied yet — the home
    # screen must never hard-fail because of a pending migration.
    try:
        sessions_res = (
            supabase.table("sessions")
            .select("id, overall_score, wpm, filler_count, started_at, mode, duration_seconds")
            .eq("user_id", user["id"])
            .order("started_at", desc=True)
            .limit(90)
            .execute()
        )
    except Exception:
        sessions_res = (
            supabase.table("sessions")
            .select("id, started_at, mode, duration_seconds")
            .eq("user_id", user["id"])
            .order("started_at", desc=True)
            .limit(90)
            .execute()
        )
    user_res = (
        supabase.table("users")
        .select("cefr_level, ielts_score, target_exam, preferred_accent")
        .eq("id", user["id"])
        .single()
        .execute()
    )
    total_sessions_res = (
        supabase.table("sessions")
        .select("id", count="exact")
        .eq("user_id", user["id"])
        .execute()
    )
    errors_fixed_res = (
        supabase.table("errors")
        .select("id", count="exact")
        .eq("user_id", user["id"])
        .eq("resolved", True)
        .execute()
    )

    all_sessions = sessions_res.data or []
    user_data = user_res.data or {}

    # Compute streak from session dates (no separate DB column needed)
    user_data["streak_days"]        = _compute_streak(all_sessions)
    user_data["total_sessions"]     = total_sessions_res.count or 0
    user_data["total_errors_fixed"] = errors_fixed_res.count or 0

    # ── Live home-card stats, computed from real scorecards ──────────────────
    # progress_percentage: average overall score of the last 10 scored sessions
    # (the natural "how close am I to the next level" proxy).
    recent_scores = [s["overall_score"] for s in all_sessions[:10] if s.get("overall_score") is not None]
    user_data["progress_percentage"] = int(round(sum(recent_scores) / len(recent_scores))) if recent_scores else 0

    # confidence_score: mean of the audio-native confidence judgment (1-10 →
    # percent) from the last 10 scorecards. Falls back to None when unscored.
    try:
        fb_res = (
            supabase.table("sessions")
            .select("feedback_json")
            .eq("user_id", user["id"])
            .not_.is_("feedback_json", "null")
            .order("started_at", desc=True)
            .limit(10)
            .execute()
        )
        confs = []
        for row in (fb_res.data or []):
            c = ((row.get("feedback_json") or {}).get("delivery") or {}).get("confidence")
            if isinstance(c, (int, float)):
                confs.append(float(c))
        user_data["confidence_score"] = int(round(sum(confs) / len(confs) * 10)) if confs else None
    except Exception:
        user_data["confidence_score"] = None

    # next_level: one CEFR step above the current level.
    ladder = ["A1", "A2", "B1", "B2", "C1", "C2"]
    cur = user_data.get("cefr_level")
    user_data["next_level"] = (
        ladder[min(ladder.index(cur) + 1, len(ladder) - 1)] if cur in ladder else None
    )

    # Return only the 10 most recent for the home/history screens
    return {
        "user": user_data,
        "recent_sessions": all_sessions[:10],
    }


@router.patch("/errors/{error_id}/resolve")
async def resolve_error(error_id: str, user=Depends(get_current_user)):
    """Mark a recurring error as resolved so it's hidden from the errors list."""
    supabase = get_supabase()
    supabase.table("errors").update({"resolved": True}).eq("id", error_id).eq("user_id", user["id"]).execute()
    return {"status": "resolved"}


@router.get("/{session_id}")
async def get_session(session_id: str, user=Depends(get_current_user)):
    supabase = get_supabase()
    res = (
        supabase.table("sessions")
        .select("*, errors(*)")
        .eq("id", session_id)
        .eq("user_id", user["id"])
        .single()
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return res.data
