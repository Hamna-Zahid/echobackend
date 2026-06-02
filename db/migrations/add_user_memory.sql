-- ── User Memory Migration ─────────────────────────────────────────────────────
-- Run this once in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- Adds persistent AI memory to each user so Echo remembers their learning journey.

-- 1. Add memory column to users table
alter table public.users
  add column if not exists memory jsonb default '{}'::jsonb;

-- 2. Add full_name (needed for personalised greetings)
alter table public.users
  add column if not exists full_name text;

-- 3. Add exam_target (used in accent-mode context routing)
alter table public.users
  add column if not exists exam_target text;

-- 4. Index so memory lookups are fast
create index if not exists idx_users_memory on public.users using gin(memory);

-- Memory JSON shape (populated by the backend after each session):
-- {
--   "session_count":        5,          -- total sessions completed
--   "total_minutes":        47,         -- cumulative practice time
--   "mastered":             ["articles", "past perfect"],
--   "persistent_struggles": ["prepositions", "subject-verb agreement"],
--   "preferred_topics":     ["technology", "career"],
--   "personality_notes":    "Speaks quickly under pressure. Responds well to encouragement.",
--   "last_session_summary": "Worked on conditional sentences. Solid progress on 'if' clauses.",
--   "last_session_mode":    "training",
--   "last_session_at":      "2026-05-10T12:34:56Z"
-- }
