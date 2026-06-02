from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from datetime import date, datetime, timezone, timedelta

from db.supabase import get_supabase, get_current_user

router = APIRouter()


def _latest_roadmap(supabase, user_id: str) -> dict:
    """
    Fetch the user's most recent learning roadmap from learning_plans.plan_json.
    Returns {} if no diagnostic has been run yet. Never raises.
    """
    try:
        res = (
            supabase.table("learning_plans")
            .select("plan_json, generated_at")
            .eq("user_id", user_id)
            .order("generated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return {}
        plan = res.data[0].get("plan_json") or {}
        roadmap = plan.get("roadmap") if isinstance(plan, dict) else None
        return roadmap if isinstance(roadmap, dict) else {}
    except Exception:
        return {}


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

    # Fetch up to 90 sessions for streak calculation + recent display
    sessions_res = (
        supabase.table("sessions")
        .select("id, fluency_score, lexical_score, grammar_score, pronunciation_score, cefr_score, started_at, mode, duration_seconds")
        .eq("user_id", user["id"])
        .order("started_at", desc=True)
        .limit(90)
        .execute()
    )
    user_res = (
        supabase.table("users")
        .select("full_name, cefr_level, ielts_score, target_exam, preferred_accent, goal")
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

    # Gamification + entitlements (single source of truth: users.memory)
    from services.user_state import get_memory
    from services.gamification import get_progress as _gami_progress
    from services.entitlements import entitlements as _entitlements
    _mem = get_memory(user["id"])
    gami = _gami_progress(user["id"], cefr=user_data.get("cefr_level"), memory=_mem)

    # Prefer the gamified streak; fall back to the computed one
    user_data["streak_days"]        = gami.get("streak_days") or _compute_streak(all_sessions)
    user_data["total_sessions"]     = total_sessions_res.count or 0
    user_data["total_errors_fixed"] = errors_fixed_res.count or 0
    # Attach the AI-generated roadmap (stored in learning_plans) for the home "Your Plan" card
    user_data["roadmap"]            = _latest_roadmap(supabase, user["id"])

    return {
        "user": user_data,
        "recent_sessions": all_sessions[:10],
        "gamification": gami,
        "entitlements": _entitlements(_mem),
    }


@router.get("/roadmap")
async def get_roadmap(user=Depends(get_current_user)):
    """
    Return the user's AI-generated learning roadmap (from the first diagnostic)
    plus the computed 'next' session to take. Empty roadmap if no diagnostic yet.
    """
    supabase = get_supabase()
    res = (
        supabase.table("users")
        .select("goal, cefr_level, ielts_score")
        .eq("id", user["id"])
        .single()
        .execute()
    )
    data = res.data or {}
    roadmap = _latest_roadmap(supabase, user["id"])
    sessions = roadmap.get("sessions") if isinstance(roadmap, dict) else None
    sessions = sessions if isinstance(sessions, list) else []
    done = int(roadmap.get("completed_count", 0)) if isinstance(roadmap, dict) else 0

    next_session = sessions[done] if 0 <= done < len(sessions) else None

    from services.gamification import get_progress as _gami
    return {
        "roadmap": roadmap,
        "goal": data.get("goal") or (roadmap.get("goal") if isinstance(roadmap, dict) else None),
        "cefr_level": data.get("cefr_level"),
        "ielts_score": data.get("ielts_score"),
        "total_sessions": len(sessions),
        "completed_count": done,
        "next_session": next_session,
        "has_plan": len(sessions) > 0,
        "gamification": _gami(user["id"], cefr=data.get("cefr_level")),
    }


@router.patch("/errors/{error_id}/resolve")
async def resolve_error(error_id: str, user=Depends(get_current_user)):
    """Mark a recurring error as resolved so it's hidden from the errors list."""
    supabase = get_supabase()
    supabase.table("errors").update({"resolved": True}).eq("id", error_id).eq("user_id", user["id"]).execute()
    return {"status": "resolved"}


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str, user=Depends(get_current_user)):
    """
    Post-session screen payload: the session row + XP/level result for this session
    (idempotently awarded) + current gamification + the AI report (if ready yet).
    """
    from datetime import datetime, timezone
    from services.user_state import get_memory
    from services.gamification import award_session, get_progress
    supabase = get_supabase()

    row = (
        supabase.table("sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user["id"])
        .single()
        .execute()
    ).data
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    # errors corrected in this session (for the report)
    errs = (
        supabase.table("errors")
        .select("error_type, original_utterance, corrected_form, explanation")
        .eq("session_id", session_id)
        .execute()
    ).data or []

    from services.entitlements import is_premium as _is_premium
    mem = get_memory(user["id"])
    premium = _is_premium(mem)
    cefr = None
    try:
        u = supabase.table("users").select("cefr_level").eq("id", user["id"]).single().execute()
        cefr = (u.data or {}).get("cefr_level")
    except Exception:
        pass

    # minutes from the row (duration filled by DB trigger on close)
    minutes = 1
    if row.get("duration_seconds"):
        minutes = max(1, int(row["duration_seconds"]) // 60)
    elif row.get("started_at") and row.get("ended_at"):
        try:
            a = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
            b = datetime.fromisoformat(row["ended_at"].replace("Z", "+00:00"))
            minutes = max(1, int((b - a).total_seconds()) // 60)
        except Exception:
            pass

    xp = award_session(user["id"], row.get("mode", "training"), minutes, cefr=cefr, session_id=session_id)
    gami = get_progress(user["id"], cefr=cefr)

    report = None
    reports = mem.get("reports")
    if isinstance(reports, dict):
        report = reports.get(session_id)
    if not report and isinstance(row.get("report_json"), dict):
        report = row["report_json"]

    return {
        "session": row,
        "errors": errs,
        "xp": xp,
        "gamification": gami,
        "is_premium": premium,
        "report": report if premium else None,   # deep report is a premium feature
        "report_ready": report is not None,
        "report_locked": (report is not None) and (not premium),
    }


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
