import os
import sys
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=logging.INFO)

from routers import auth, sessions, ws, echo, admin
from db.supabase import init_supabase


def _validate_env():
    required = {
        "GROQ_API_KEY":              "Free key at console.groq.com (Whisper STT + Llama 3.3 70B)",
        "SUPABASE_URL":              "supabase.com → project → Settings → API",
        "SUPABASE_ANON_KEY":         "supabase.com → project → Settings → API",
        "SUPABASE_SERVICE_ROLE_KEY": "supabase.com → project → Settings → API",
    }
    missing = [f"  {var}  ← {hint}" for var, hint in required.items() if not os.environ.get(var)]
    if missing:
        print("\n" + "═" * 60)
        print("  Echo: missing required environment variables")
        print("═" * 60)
        for m in missing:
            print(m)
        print("\n  Copy .env.example → .env and fill in the values.")
        print("═" * 60 + "\n")
        sys.exit(1)

    tts = os.environ.get("DEV_TTS", "edge").lower()
    print(f"[Echo] Cascade: Groq Whisper + Llama 3.3 70B + Edge TTS ({tts}) — free tier, no GPU")
    if os.environ.get("GEMINI_API_KEY"):
        print(f"[Echo] Gemma 4 audio delivery analysis: ON ({os.environ.get('GEMMA_AUDIO_MODEL', 'gemma-4-12b-it')})")
    else:
        print("[Echo] Gemma 4 audio delivery analysis: off (set GEMINI_API_KEY to enable — free at aistudio.google.com)")


_validate_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_supabase()
    yield


app = FastAPI(title="Echo API", version="2.0.0", lifespan=lifespan)

# CORS only matters for browser clients (the /admin dashboard is same-origin;
# the mobile app is native and unaffected). Lock to specific origins in prod:
#   ALLOWED_ORIGINS=https://admin.example.com,https://echo.example.com
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router,     prefix="/auth",     tags=["auth"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(echo.router,     tags=["echo-test"])
app.include_router(ws.router,       tags=["websocket"])
app.include_router(admin.router,    tags=["admin"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Echo API"}
