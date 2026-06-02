"""
Gamification — Duolingo-style XP, CEFR-aligned levels, streaks, daily goals, leagues.

State lives in users.memory["gamification"]:
  {
    "level": 11,                 # 1..30, maps to a CEFR sub-stage (B1.1 ...)
    "xp": 40,                    # XP earned within the current level
    "total_xp": 1240,            # lifetime XP (drives league)
    "streak_days": 5,
    "longest_streak": 9,
    "last_practice_date": "2026-06-01",
    "daily_xp": 20,
    "daily_goal_xp": 30,
    "daily_date": "2026-06-01",
    "sessions_completed": 12
  }

Levels 1..30 map to CEFR sub-stages A1.1 .. C2.5 (5 sub-stages per band).
"""

from datetime import date, datetime, timezone, timedelta

from services.user_state import update_memory, get_memory

BANDS = ["A1", "A2", "B1", "B2", "C1", "C2"]
MAX_LEVEL = 30
DEFAULT_DAILY_GOAL = 30

CEFR_BASE_LEVEL = {"A1": 1, "A2": 6, "B1": 11, "B2": 16, "C1": 21, "C2": 26}

LEAGUE_BY_TIER = ["Bronze", "Silver", "Gold", "Sapphire", "Ruby", "Diamond"]


# ── Pure helpers ──────────────────────────────────────────────────────────────

def cefr_substage(level: int) -> str:
    """Level 1..30 → 'A1.1' .. 'C2.5'."""
    level = max(1, min(MAX_LEVEL, int(level)))
    band = BANDS[(level - 1) // 5]
    sub = ((level - 1) % 5) + 1
    return f"{band}.{sub}"


def league_for_level(level: int) -> str:
    level = max(1, min(MAX_LEVEL, int(level)))
    return LEAGUE_BY_TIER[(level - 1) // 5]


def xp_to_next(level: int) -> int:
    """XP required to advance FROM `level` to level+1 (rising curve)."""
    return 120 + (max(1, int(level)) - 1) * 15


def xp_for_session(mode: str, minutes: int) -> int:
    """How much XP a finished session is worth."""
    base = 60 if mode == "diagnostic" else 25
    return int(base + min(max(0, minutes), 15) * 2)


def _level_from_cefr(cefr: str | None) -> int:
    if not cefr:
        return 1
    return CEFR_BASE_LEVEL.get(str(cefr).upper().strip(), 1)


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ── Seeding & defaults ────────────────────────────────────────────────────────

def _default_gamification(cefr: str | None = None) -> dict:
    return {
        "level": _level_from_cefr(cefr),
        "xp": 0,
        "total_xp": 0,
        "streak_days": 0,
        "longest_streak": 0,
        "last_practice_date": None,
        "daily_xp": 0,
        "daily_goal_xp": DEFAULT_DAILY_GOAL,
        "daily_date": None,
        "sessions_completed": 0,
    }


def _ensure(mem: dict, cefr: str | None = None) -> dict:
    g = mem.get("gamification")
    if not isinstance(g, dict) or "level" not in g:
        g = _default_gamification(cefr)
        mem["gamification"] = g
    return g


# ── Award XP at the end of a session ─────────────────────────────────────────

def award_session(user_id: str, mode: str, minutes: int, cefr: str | None = None,
                  session_id: str | None = None) -> dict:
    """
    Award XP for a completed session, update streak + daily goal + level.
    Idempotent by session_id: a session is only ever counted once, so it's safe
    to call from both the WebSocket close and the HTTP session-summary endpoint.
    Returns a summary the client can celebrate with. Best-effort.
    """
    gained = xp_for_session(mode, minutes)
    result: dict = {}

    def mutate(mem: dict):
        g = _ensure(mem, cefr)
        today = _today()
        today_s = today.isoformat()

        # Idempotency — don't double-award the same session.
        awarded = g.get("awarded_sessions")
        if not isinstance(awarded, list):
            awarded = []
            g["awarded_sessions"] = awarded
        if session_id and session_id in awarded:
            prev = g.get("last_award") or {}
            if prev.get("session_id") == session_id:
                result.update(prev)
            else:
                # awarded earlier but not the most recent — return a no-gain snapshot
                result.update({
                    "session_id": session_id, "xp_gained": 0, "leveled_up": False,
                    "level": g["level"], "cefr_stage": cefr_substage(g["level"]),
                    "league": league_for_level(g["level"]), "streak_days": g.get("streak_days", 0),
                    "daily_xp": g.get("daily_xp", 0), "daily_goal_xp": g.get("daily_goal_xp", DEFAULT_DAILY_GOAL),
                    "daily_goal_met": g.get("daily_xp", 0) >= g.get("daily_goal_xp", DEFAULT_DAILY_GOAL),
                    "xp": g.get("xp", 0), "xp_to_next": xp_to_next(g["level"]), "already_awarded": True,
                })
            return

        # ── streak ──
        last = g.get("last_practice_date")
        if last != today_s:
            if last == (today - timedelta(days=1)).isoformat():
                g["streak_days"] = int(g.get("streak_days", 0)) + 1
            else:
                g["streak_days"] = 1
            g["last_practice_date"] = today_s
        g["longest_streak"] = max(int(g.get("longest_streak", 0)), int(g.get("streak_days", 0)))

        # ── daily goal ──
        if g.get("daily_date") != today_s:
            g["daily_xp"] = 0
            g["daily_date"] = today_s
        g["daily_xp"] = int(g.get("daily_xp", 0)) + gained

        # ── XP + level up ──
        g["total_xp"] = int(g.get("total_xp", 0)) + gained
        g["xp"] = int(g.get("xp", 0)) + gained
        leveled_up = False
        new_levels = 0
        while g["xp"] >= xp_to_next(g["level"]) and g["level"] < MAX_LEVEL:
            g["xp"] -= xp_to_next(g["level"])
            g["level"] += 1
            leveled_up = True
            new_levels += 1
        if g["level"] >= MAX_LEVEL:
            g["level"] = MAX_LEVEL

        g["sessions_completed"] = int(g.get("sessions_completed", 0)) + 1

        result.update({
            "session_id": session_id,
            "xp_gained": gained,
            "level": g["level"],
            "leveled_up": leveled_up,
            "levels_gained": new_levels,
            "cefr_stage": cefr_substage(g["level"]),
            "league": league_for_level(g["level"]),
            "streak_days": g["streak_days"],
            "daily_xp": g["daily_xp"],
            "daily_goal_xp": g.get("daily_goal_xp", DEFAULT_DAILY_GOAL),
            "daily_goal_met": g["daily_xp"] >= g.get("daily_goal_xp", DEFAULT_DAILY_GOAL),
            "xp": g["xp"],
            "xp_to_next": xp_to_next(g["level"]),
        })

        # Record idempotency + remember the most recent award for the summary screen.
        if session_id:
            awarded.append(session_id)
            g["awarded_sessions"] = awarded[-50:]
            g["last_award"] = dict(result)

    update_memory(user_id, mutate)
    return result


# ── Read progress for the UI ─────────────────────────────────────────────────

def get_progress(user_id: str, cefr: str | None = None, memory: dict | None = None) -> dict:
    """Return the gamification summary for the home/path UI. Never raises."""
    mem = memory if isinstance(memory, dict) else get_memory(user_id)
    g = mem.get("gamification")
    if not isinstance(g, dict) or "level" not in g:
        g = _default_gamification(cefr)

    today_s = _today().isoformat()
    daily_xp = int(g.get("daily_xp", 0)) if g.get("daily_date") == today_s else 0
    streak = int(g.get("streak_days", 0))
    # Streak is "active" only if practiced today or yesterday
    last = g.get("last_practice_date")
    if last not in (today_s, (_today() - timedelta(days=1)).isoformat()):
        streak_display = 0 if last is None else streak  # keep historical streak but UI may grey it
    else:
        streak_display = streak

    level = int(g.get("level", 1))
    goal = int(g.get("daily_goal_xp", DEFAULT_DAILY_GOAL))
    return {
        "level": level,
        "cefr_stage": cefr_substage(level),
        "league": league_for_level(level),
        "xp": int(g.get("xp", 0)),
        "xp_to_next": xp_to_next(level),
        "total_xp": int(g.get("total_xp", 0)),
        "streak_days": streak_display,
        "longest_streak": int(g.get("longest_streak", 0)),
        "daily_xp": daily_xp,
        "daily_goal_xp": goal,
        "daily_goal_met": daily_xp >= goal,
        "sessions_completed": int(g.get("sessions_completed", 0)),
    }
