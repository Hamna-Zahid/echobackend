-- ── Learning Roadmap Migration ───────────────────────────────────────────────
-- Run once in Supabase SQL Editor (Dashboard → SQL Editor → New query).
-- Adds the AI-generated multi-session learning plan produced by the first
-- goal-discovery diagnostic session.

alter table public.users
  add column if not exists roadmap jsonb default '{}'::jsonb;

-- Roadmap JSON shape (written by the backend after the first diagnostic):
-- {
--   "goal":        "IELTS Band 7.0",                 -- discovered conversationally
--   "motivation":  "Applying for a UK master's in autumn",
--   "timeline":    "about 3 months",
--   "completed_count": 0,                            -- how many plan sessions are done
--   "sessions": [
--     {
--       "n": 1,
--       "title": "Past & Present Perfect Foundations",
--       "mode": "training",                          -- diagnostic|training|interview|conversation|professional|accent
--       "accent": "american",
--       "focus": "tense accuracy",
--       "description": "Narrate past experiences to lock in present perfect vs simple past.",
--       "skills": ["present perfect", "simple past", "time markers"],
--       "duration_min": 10
--     }
--     // ... 5–8 sessions total
--   ]
-- }
