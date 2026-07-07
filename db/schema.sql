-- EchoCoach Supabase Schema
-- Apply via Supabase SQL Editor. Enable RLS on all tables.

-- Users (extends Supabase auth.users)
create table public.users (
  id uuid primary key references auth.users(id) on delete cascade,
  cefr_level text,
  ielts_score decimal(3,1),
  target_exam text check (target_exam in ('IELTS','TOEFL','INTERVIEW')),
  preferred_accent text default 'american' check (preferred_accent in ('american','british')),
  goal text,
  streak_days int default 0,
  created_at timestamptz default now(),
  last_active timestamptz default now()
);

-- Sessions
create table public.sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.users(id) on delete cascade,
  started_at timestamptz default now(),
  ended_at timestamptz,
  duration_seconds int,
  mode text check (mode in ('diagnostic','training','interview','conversation','professional','accent')),
  cefr_score text,
  fluency_score decimal(3,1),
  lexical_score decimal(3,1),
  grammar_score decimal(3,1),
  pronunciation_score decimal(3,1),
  transcript text,
  created_at timestamptz default now()
);

-- Errors — the personalization engine backbone
create table public.errors (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.users(id) on delete cascade,
  session_id uuid references public.sessions(id) on delete cascade,
  error_type text not null, -- TENSE | ARTICLE | PREPOSITION | FILLER | COHERENCE | VOCAB
  original_utterance text not null,
  corrected_form text not null,
  explanation text,
  recurrence_count int default 1,
  resolved boolean default false,
  last_seen timestamptz default now(),
  created_at timestamptz default now()
);

-- Learning plans
create table public.learning_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.users(id) on delete cascade,
  generated_at timestamptz default now(),
  plan_json jsonb not null -- stores full diagnostic output
);

-- RLS policies
alter table public.users enable row level security;
alter table public.sessions enable row level security;
alter table public.errors enable row level security;
alter table public.learning_plans enable row level security;

create policy "Users own their data" on public.users for all using (auth.uid() = id);
create policy "Users own their sessions" on public.sessions for all using (auth.uid() = user_id);
create policy "Users own their errors" on public.errors for all using (auth.uid() = user_id);
create policy "Users own their plans" on public.learning_plans for all using (auth.uid() = user_id);

-- Indexes for performance
create index idx_sessions_user_id on public.sessions(user_id);
create index idx_sessions_started_at on public.sessions(started_at desc);
create index idx_errors_user_id on public.errors(user_id);
create index idx_errors_recurrence on public.errors(user_id, recurrence_count desc);
create index idx_errors_type on public.errors(user_id, error_type);

-- Function: auto-update last_active on users
create or replace function public.update_user_last_active()
returns trigger language plpgsql security definer as $$
begin
  update public.users set last_active = now() where id = new.user_id;
  return new;
end;
$$;

create trigger sessions_update_last_active
  after insert on public.sessions
  for each row execute function public.update_user_last_active();

-- Function: compute session duration on close
create or replace function public.compute_session_duration()
returns trigger language plpgsql as $$
begin
  if new.ended_at is not null and old.ended_at is null then
    new.duration_seconds := extract(epoch from (new.ended_at - new.started_at))::int;
  end if;
  return new;
end;
$$;

create trigger sessions_compute_duration
  before update on public.sessions
  for each row execute function public.compute_session_duration();
