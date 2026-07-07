# Echo Backend

FastAPI backend for Echo — a voice-based AI English coaching app. Single adaptive
coaching mode (no separate IELTS/interview/daily modes): live voice conversation
over WebSocket, with a structured feedback scorecard generated after each session.

## Stack (all free tier, no GPU required)

- **STT**: Groq Whisper Large-v3
- **LLM**: Groq Llama 3.3 70B (conversation + JSON feedback scoring)
- **TTS**: Microsoft Edge Neural TTS (free, no key)
- **Audio delivery analysis** (optional): Gemma 4 12B via Google AI Studio — hears
  tone/confidence/hesitation from the actual recordings, feeds live coaching tips
  + the scorecard
- **Auth + DB**: Supabase (Postgres + RLS)
- **Admin panel**: built-in, served at `/admin`

## Local setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows; source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env         # fill in Supabase + Groq keys
uvicorn main:app --reload
```

`curl http://localhost:8000/health` → `{"status":"ok","service":"Echo API"}`

## Required environment variables

See `.env.example` for the full list. Minimum to boot: `GROQ_API_KEY`,
`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.
`GEMINI_API_KEY` is optional (enables Gemma 4 audio delivery analysis).

## Database

Run these in the Supabase SQL editor, in order:
1. `db/schema.sql`
2. `db/migrations/add_user_memory.sql`
3. `db/migrations/add_admin.sql`
4. `db/migrations/single_mode.sql` — **required**, drops the old mode CHECK
   constraint so the single "coach" mode can be used.

## Deploy (Render)

- Root directory: repo root (this is a backend-only repo)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set the env vars above in the Render dashboard.

Render's free tier sleeps after 15 min idle; the first request after sleep takes
~50s to wake the service.

## Admin panel

`GET /admin` — sign in with a Supabase account whose email is listed in
`ADMIN_EMAILS`, or with `users.is_admin = true`. Shows users, sessions,
transcripts, scorecards, and a health/events view (STT/LLM/feedback error rates
and latency).
