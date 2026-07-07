"""
Admin panel API — end-to-end visibility for the operator.

Everything an early-stage team needs: who your users are, what they're doing,
which sessions ran, the scores they got, and — crucially — what's working and
what's failing (the event/health view, fed by services/events.py).

Auth: any logged-in Supabase user can hit these only if they're an admin, where
"admin" = email in the ADMIN_EMAILS env allowlist OR users.is_admin = true.
All data is read with a service-role client, so it bypasses RLS and sees
everything. The web dashboard (GET /admin) calls these with the admin's JWT.
"""

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from db.supabase import get_admin_client, get_current_user

router = APIRouter()

ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}


# ── Auth ──────────────────────────────────────────────────────────────────────

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    email = (user.get("email") or "").lower()
    if email and email in ADMIN_EMAILS:
        return user
    try:
        res = get_admin_client().table("users").select("is_admin").eq("id", user["id"]).single().execute()
        if res.data and res.data.get("is_admin"):
            return user
    except Exception:
        pass
    raise HTTPException(status_code=403, detail="Admin access required")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _count(sb, table: str, since: Optional[datetime] = None, col: str = "created_at") -> int:
    try:
        q = sb.table(table).select("id", count="exact")
        if since:
            q = q.gte(col, since.isoformat())
        return q.execute().count or 0
    except Exception:
        return 0


def _email_map(sb) -> dict:
    """Map of user_id -> email, pulled from Supabase auth (service role)."""
    try:
        res = sb.auth.admin.list_users()
        users = res if isinstance(res, list) else getattr(res, "users", []) or []
        return {u.id: getattr(u, "email", None) for u in users}
    except Exception:
        return {}


# ── Overview KPIs ─────────────────────────────────────────────────────────────

@router.get("/admin/api/overview")
async def overview(admin: dict = Depends(require_admin)):
    sb = get_admin_client()
    now = _now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    d1, d7, d30 = now - timedelta(days=1), now - timedelta(days=7), now - timedelta(days=30)

    # Recent sessions (one fetch, many metrics).
    sessions = (
        sb.table("sessions")
        .select("user_id, started_at, ended_at, mode, duration_seconds, overall_score")
        .gte("started_at", d30.isoformat())
        .order("started_at", desc=True)
        .limit(5000)
        .execute().data or []
    )

    def active_since(cut):
        return len({s["user_id"] for s in sessions if (_parse_dt(s["started_at"]) or now) >= cut})

    durations = [s["duration_seconds"] for s in sessions if s.get("duration_seconds")]
    scores = [s["overall_score"] for s in sessions if s.get("overall_score") is not None]
    completed = sum(1 for s in sessions if s.get("ended_at"))
    mode_breakdown = Counter(s.get("mode") or "unknown" for s in sessions)

    # Event error rate (last 24h).
    events = (
        sb.table("app_events").select("status")
        .gte("created_at", d1.isoformat()).limit(5000).execute().data or []
    )
    err = sum(1 for e in events if e.get("status") == "error")
    error_rate = round(err / len(events) * 100, 1) if events else 0.0

    return {
        "users": {
            "total": _count(sb, "users"),
            "new_today": _count(sb, "users", today),
            "new_7d": _count(sb, "users", d7),
            "new_30d": _count(sb, "users", d30),
            "dau": active_since(d1),
            "wau": active_since(d7),
            "mau": active_since(d30),
        },
        "sessions": {
            "total": _count(sb, "sessions"),
            "today": _count(sb, "sessions", today, col="started_at"),
            "last_30d": len(sessions),
            "completion_rate": round(completed / len(sessions) * 100, 1) if sessions else 0.0,
            "avg_duration_sec": round(sum(durations) / len(durations)) if durations else 0,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        },
        "health": {
            "events_24h": len(events),
            "errors_24h": err,
            "error_rate_pct": error_rate,
        },
        "mode_breakdown": dict(mode_breakdown),
        "generated_at": now.isoformat(),
    }


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/admin/api/users")
async def users_list(
    search: str = Query(""),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    admin: dict = Depends(require_admin),
):
    sb = get_admin_client()
    rows = (
        sb.table("users").select("*").order("created_at", desc=True).limit(2000).execute().data or []
    )
    emails = _email_map(sb)

    # Session stats per user (single fetch, aggregate in Python).
    sess = sb.table("sessions").select("user_id, started_at, overall_score").limit(10000).execute().data or []
    stats: dict = {}
    for s in sess:
        uid = s["user_id"]
        st = stats.setdefault(uid, {"count": 0, "last": None, "scores": []})
        st["count"] += 1
        if not st["last"] or (s["started_at"] or "") > st["last"]:
            st["last"] = s["started_at"]
        if s.get("overall_score") is not None:
            st["scores"].append(s["overall_score"])

    enriched = []
    for r in rows:
        uid = r["id"]
        st = stats.get(uid, {"count": 0, "last": None, "scores": []})
        enriched.append({
            "id": uid,
            "email": emails.get(uid),
            "created_at": r.get("created_at"),
            "last_active": r.get("last_active"),
            "streak_days": r.get("streak_days"),
            "is_admin": r.get("is_admin", False),
            "session_count": st["count"],
            "last_session": st["last"],
            "avg_score": round(sum(st["scores"]) / len(st["scores"]), 1) if st["scores"] else None,
        })

    if search:
        q = search.lower()
        enriched = [u for u in enriched if q in (u["email"] or "").lower() or q in u["id"].lower()]

    total = len(enriched)
    return {"total": total, "users": enriched[offset:offset + limit]}


@router.get("/admin/api/users/{uid}")
async def user_detail(uid: str, admin: dict = Depends(require_admin)):
    sb = get_admin_client()
    profile = {}
    try:
        res = sb.table("users").select("*").eq("id", uid).single().execute()
        profile = res.data or {}
    except Exception:
        pass
    emails = _email_map(sb)
    sessions = (
        sb.table("sessions").select("id, mode, started_at, ended_at, duration_seconds, overall_score, wpm, filler_count")
        .eq("user_id", uid).order("started_at", desc=True).limit(100).execute().data or []
    )
    events = (
        sb.table("app_events").select("*").eq("user_id", uid)
        .order("created_at", desc=True).limit(100).execute().data or []
    )
    return {"id": uid, "email": emails.get(uid), "profile": profile, "sessions": sessions, "events": events}


# ── Sessions ──────────────────────────────────────────────────────────────────

@router.get("/admin/api/sessions")
async def sessions_list(
    mode: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    admin: dict = Depends(require_admin),
):
    sb = get_admin_client()
    q = (
        sb.table("sessions")
        .select("id, user_id, mode, started_at, ended_at, duration_seconds, overall_score, wpm, filler_count")
        .order("started_at", desc=True)
    )
    if mode:
        q = q.eq("mode", mode)
    rows = q.range(offset, offset + limit - 1).execute().data or []
    emails = _email_map(sb)
    for r in rows:
        r["email"] = emails.get(r["user_id"])
    return {"sessions": rows}


@router.get("/admin/api/sessions/{sid}")
async def session_detail(sid: str, admin: dict = Depends(require_admin)):
    sb = get_admin_client()
    session = {}
    try:
        res = sb.table("sessions").select("*").eq("id", sid).single().execute()
        session = res.data or {}
    except Exception:
        pass
    events = (
        sb.table("app_events").select("*").eq("session_id", sid)
        .order("created_at", desc=True).limit(100).execute().data or []
    )
    emails = _email_map(sb)
    if session.get("user_id"):
        session["email"] = emails.get(session["user_id"])
    return {"session": session, "events": events}


# ── Events / health ───────────────────────────────────────────────────────────

@router.get("/admin/api/events")
async def events_list(
    status: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    admin: dict = Depends(require_admin),
):
    sb = get_admin_client()
    q = sb.table("app_events").select("*").order("created_at", desc=True)
    if status:
        q = q.eq("status", status)
    if event_type:
        q = q.eq("event_type", event_type)
    rows = q.limit(limit).execute().data or []
    emails = _email_map(sb)
    for r in rows:
        if r.get("user_id"):
            r["email"] = emails.get(r["user_id"])
    return {"events": rows}


@router.get("/admin/api/health")
async def health(admin: dict = Depends(require_admin)):
    sb = get_admin_client()
    d1 = _now() - timedelta(days=1)
    events = (
        sb.table("app_events").select("event_type, status, latency_ms")
        .gte("created_at", d1.isoformat()).limit(10000).execute().data or []
    )
    stages: dict = {}
    for e in events:
        et = e.get("event_type", "unknown")
        s = stages.setdefault(et, {"total": 0, "errors": 0, "latencies": []})
        s["total"] += 1
        if e.get("status") == "error":
            s["errors"] += 1
        if e.get("latency_ms") is not None:
            s["latencies"].append(e["latency_ms"])
    out = {}
    for et, s in stages.items():
        lat = s["latencies"]
        out[et] = {
            "total": s["total"],
            "errors": s["errors"],
            "error_rate_pct": round(s["errors"] / s["total"] * 100, 1) if s["total"] else 0.0,
            "avg_latency_ms": round(sum(lat) / len(lat)) if lat else None,
        }
    return {"window": "24h", "stages": out}


@router.get("/admin/api/timeseries")
async def timeseries(days: int = Query(30, le=90), admin: dict = Depends(require_admin)):
    sb = get_admin_client()
    since = _now() - timedelta(days=days)
    users = sb.table("users").select("created_at").gte("created_at", since.isoformat()).limit(10000).execute().data or []
    sessions = sb.table("sessions").select("started_at").gte("started_at", since.isoformat()).limit(20000).execute().data or []

    labels = [(since + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
    signups = {d: 0 for d in labels}
    sess = {d: 0 for d in labels}
    for u in users:
        dt = _parse_dt(u.get("created_at"))
        if dt:
            signups[dt.strftime("%Y-%m-%d")] = signups.get(dt.strftime("%Y-%m-%d"), 0) + 1
    for s in sessions:
        dt = _parse_dt(s.get("started_at"))
        if dt:
            sess[dt.strftime("%Y-%m-%d")] = sess.get(dt.strftime("%Y-%m-%d"), 0) + 1
    return {
        "labels": labels,
        "signups": [signups.get(d, 0) for d in labels],
        "sessions": [sess.get(d, 0) for d in labels],
    }


# ── Dashboard HTML ────────────────────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the single-file admin dashboard, with Supabase public config injected."""
    html_path = Path(__file__).parent.parent / "admin" / "dashboard.html"
    try:
        html = html_path.read_text(encoding="utf-8")
    except Exception:
        return HTMLResponse("<h1>Admin dashboard file missing</h1>", status_code=500)
    html = (
        html.replace("__SUPABASE_URL__", os.environ.get("SUPABASE_URL", ""))
            .replace("__SUPABASE_ANON_KEY__", os.environ.get("SUPABASE_ANON_KEY", ""))
    )
    return HTMLResponse(html)
