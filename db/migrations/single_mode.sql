-- Single-mode migration: the app now runs one adaptive "coach" mode.
-- The original sessions.mode CHECK constraint only allowed the old mode list,
-- which would reject 'coach' inserts. Drop it and allow any mode string.
-- Apply via Supabase SQL Editor (safe to run once).

alter table public.sessions drop constraint if exists sessions_mode_check;

-- Keep users.cefr_level fresh: it is now updated automatically by the backend
-- after each session (feedback engine estimates CEFR from the transcript).
-- No schema change needed for that — documented here for clarity.
