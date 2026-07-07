-- Admin panel + observability migration.
-- Apply via Supabase SQL Editor (safe to run once; uses IF NOT EXISTS).

-- 1) Persist the structured feedback scorecard on each session.
alter table public.sessions add column if not exists overall_score int;
alter table public.sessions add column if not exists wpm int;
alter table public.sessions add column if not exists filler_count int;
alter table public.sessions add column if not exists feedback_json jsonb;

-- 2) Admin flag (in addition to the ADMIN_EMAILS env allowlist).
alter table public.users add column if not exists is_admin boolean default false;

-- 3) Event log — powers the "what's working / what's not" view.
--    Every important backend event is recorded here: session lifecycle, and the
--    STT / LLM / feedback stages with latency + ok|error status.
create table if not exists public.app_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  session_id uuid,
  event_type text not null,         -- session_started | stt | llm | feedback | session_completed | ...
  status text not null default 'ok',-- ok | error
  latency_ms int,
  detail jsonb,
  created_at timestamptz default now()
);

create index if not exists idx_app_events_created on public.app_events(created_at desc);
create index if not exists idx_app_events_type on public.app_events(event_type, status);
create index if not exists idx_app_events_user on public.app_events(user_id);

-- Only the service role (the backend) reads/writes events. Enable RLS with no
-- public policy so it is locked down to server-side access.
alter table public.app_events enable row level security;

-- Helpful index for admin session listing by score.
create index if not exists idx_sessions_overall_score on public.sessions(overall_score desc);
