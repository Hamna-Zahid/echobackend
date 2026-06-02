import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

from routers import auth, sessions, ws, echo, billing
from db.supabase import init_supabase


def _validate_env():
    dev_tts = os.environ.get("DEV_TTS", "edge").lower()

    required = {
        "GEMINI_API_KEY":            "Get free key at aistudio.google.com → Get API Key",
        "GROQ_API_KEY":              "Get free key at console.groq.com (used for Whisper STT + LLM fallback)",
        "SUPABASE_URL":              "Get from supabase.com → project → Settings → API",
        "SUPABASE_ANON_KEY":         "Get from supabase.com → project → Settings → API",
        "SUPABASE_SERVICE_ROLE_KEY": "Get from supabase.com → project → Settings → API",
        "REDIS_URL":                 "Get from console.upstash.com (free tier)",
    }

    if dev_tts == "cartesia":
        required["CARTESIA_API_KEY"]           = "Get from play.cartesia.ai"
        required["CARTESIA_VOICE_ID_AMERICAN"] = "Pick a voice ID from Cartesia voice library"

    missing = [
        f"  {var}  ← {hint}"
        for var, hint in required.items()
        if not os.environ.get(var)
    ]

    if missing:
        print("\n" + "═" * 60)
        print("  EchoCoach: missing required environment variables")
        print("═" * 60)
        for m in missing:
            print(m)
        print(f"\n  Copy .env.example → .env and fill in the values.")
        print(f"  Current DEV_TTS mode: {dev_tts}")
        print("═" * 60 + "\n")
        sys.exit(1)

    if dev_tts == "kokoro":
        print("[EchoCoach] TTS mode: Kokoro-82M (FREE — local neural TTS, CPU)")
    elif dev_tts == "edge":
        print("[EchoCoach] TTS mode: edge-tts (FREE — Microsoft Edge Neural voices)")
    elif dev_tts == "silent":
        print("[EchoCoach] TTS mode: silent (no audio — AI logic testing only)")
    elif dev_tts == "cartesia":
        print("[EchoCoach] TTS mode: Cartesia Sonic (production)")
    else:
        print(f"[EchoCoach] TTS mode: {dev_tts}")


_validate_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_supabase()
    yield


app = FastAPI(
    title="EchoCoach API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # dev: allow all origins (phone LAN IPs vary)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _log_startup() -> None:
    import socket
    try:
        # Print the LAN IP so the developer knows what to put in mobile/.env
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "unknown"
    print("\n" + "=" * 60)
    print("  EchoCoach API is running")
    print(f"  Local:   http://127.0.0.1:8765")
    print(f"  Network: http://{lan_ip}:8765  ← use this in mobile/.env")
    print(f"  Emulator:http://10.0.2.2:8765  ← Android emulator alias")
    print("=" * 60 + "\n")

app.include_router(auth.router,     prefix="/auth",     tags=["auth"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(billing.router,  prefix="/billing",  tags=["billing"])
app.include_router(echo.router,     tags=["echo-test"])
app.include_router(ws.router,       tags=["websocket"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "EchoCoach API"}
