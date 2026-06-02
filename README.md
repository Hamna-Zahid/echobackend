# EchoCoach — Backend

FastAPI + async WebSockets backend for EchoCoach, an AI English-coaching app.
Real-time voice coaching: Groq Whisper (STT) → Gemini/Groq (LLM) → edge-tts (TTS),
with Supabase (auth + DB) and Upstash Redis (live session state).

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt   # full set incl. post-session phoneme analysis
cp .env.example .env              # then fill in your keys
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `GET /health` → `{"status":"ok"}`.

## Deploy (Railway, Docker)
This repo deploys as-is — `Dockerfile` is at the root (leave Railway's
**Root Directory** at default). Uses the lightweight `requirements-deploy.txt`
(no PyTorch) for a fast ~510 MB image.

Set these environment variables in the Railway dashboard:

| Variable | Notes |
|---|---|
| `GEMINI_API_KEY` | aistudio.google.com |
| `GROQ_API_KEY` | console.groq.com (Whisper STT + LLM) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key (server-only) |
| `REDIS_URL` | Upstash `rediss://...` |
| `DEV_TTS` | `edge` |

The container validates these on boot and exits if any are missing.

## Notes
- Secrets live only in `.env` (gitignored) and the host's env — never committed.
- `requirements.txt` = full local dev (adds torch/transformers/kokoro for phoneme
  analysis). `requirements-deploy.txt` = lean runtime used by the Docker image.
