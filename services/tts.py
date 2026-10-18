"""
TTS service — streaming voice synthesis.

Engines (selected by DEV_TTS / TTS_ENGINE / TTS_PRIMARY):
  edge      → Microsoft Edge Neural TTS (default — FREE, no key, good quality)
  piper     → Local ONNX (Piper) — very low first-byte latency, offline after
              the voice model is downloaded once
  sapi      → Windows SAPI (offline, no network)
  cartesia  → Cartesia Sonic streaming (needs CARTESIA_API_KEY / voice ids)
  silent    → 0.3 s of silence per call (AI-logic testing only, fastest)

Always yields raw PCM F32LE chunks at 24 kHz (what the mobile client plays).
A small phrase cache keeps repeated coach lines free.
"""

import base64
import io
import json
import os
import struct
import asyncio
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import httpx

EDGE_VOICES = {
    "american": "en-US-AriaNeural",
    "british":  "en-GB-SoniaNeural",
    "male":     "en-US-GuyNeural",
}

CARTESIA_URL = "https://api.cartesia.ai/tts/sse"
CARTESIA_VOICES = {
    "american": os.environ.get("CARTESIA_VOICE_ID_AMERICAN", ""),
    "british":  os.environ.get("CARTESIA_VOICE_ID_BRITISH", ""),
}

# DEV_TTS keeps backward-compat; TTS_ENGINE/TTS_PRIMARY take priority when set.
_TTS_MODE = os.environ.get("DEV_TTS", "edge").lower()
_TTS_ENGINE = os.environ.get("TTS_ENGINE", os.environ.get("TTS_PRIMARY", "")).lower()

# ── Phrase cache ──────────────────────────────────────────────────────────────
_TTS_CACHE: dict[str, List[bytes]] = {}
_MAX_TTS_CACHE = 64

# ── Piper (ONNX) config ───────────────────────────────────────────────────────
PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "./voices/piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "en_US-amy-medium.onnx")
PIPER_MODEL_JSON = os.environ.get("PIPER_MODEL_JSON", "en_US-amy-medium.onnx.json")
PIPER_SAMPLE_RATE = int(os.environ.get("PIPER_SAMPLE_RATE", "24000"))
# Base URL of the rhasspy voice repo for the configured voice.
_PIPER_BASE_URL = os.environ.get(
    "PIPER_BASE_URL",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium",
)

_piper_voice = None  # cached PiperVoice instance


def _cache_key(text: str, accent: Optional[str]) -> str:
    s = (text or "").strip().lower() + "|" + (accent or "default")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _chunk_bytes(pcm: bytes, size: int = 16384) -> List[bytes]:
    if not pcm:
        return []
    return [pcm[i:i + size] for i in range(0, len(pcm), size)]


def _store_cache(key: str, chunks: List[bytes], text: str) -> None:
    if not chunks or len(text) > 300:
        return
    if len(_TTS_CACHE) >= _MAX_TTS_CACHE:
        _TTS_CACHE.pop(next(iter(_TTS_CACHE)), None)
    _TTS_CACHE[key] = list(chunks)


# ── Public entrypoint ─────────────────────────────────────────────────────────

async def stream_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """Stream PCM F32LE audio chunks for `text`. Caches repeated phrases."""
    if not text or not text.strip():
        return

    if _TTS_MODE == "silent":
        yield bytes(int(sample_rate * 0.3) * 4)
        return

    engine = (_TTS_ENGINE or _TTS_MODE or "edge").lower()

    cache_k = _cache_key(text, accent)
    cached = _TTS_CACHE.get(cache_k)
    if cached:
        for chunk in cached:
            yield chunk
        return

    collected: List[bytes] = []

    # 1) Piper — local ONNX, lowest latency. Falls through if unavailable.
    if engine in ("piper",):
        async for chunk in _piper_tts(text, accent, sample_rate):
            collected.append(chunk)
            yield chunk
        if collected:
            _store_cache(cache_k, collected, text)
            return

    # 2) Explicit SAPI/offline request.
    if engine in ("sapi", "windows", "offline"):
        async for chunk in _sapi_tts(text, accent, sample_rate):
            collected.append(chunk)
            yield chunk
        if collected:
            _store_cache(cache_k, collected, text)
        return

    # 3) Cartesia (if configured), else fall through to Edge.
    if engine == "cartesia" or _TTS_MODE == "cartesia":
        produced = False
        async for chunk in _cartesia_tts(text, accent, sample_rate):
            produced = True
            collected.append(chunk)
            yield chunk
        if collected:
            _store_cache(cache_k, collected, text)
        if produced:
            return

    # 4) Default: Edge neural TTS, fall back to offline Windows SAPI on failure.
    # (default) Edge — graceful offline fallback to Windows SAPI if Edge is
    # blocked or produces nothing (e.g. no network in the container).
    async for chunk in _edge_tts(text, accent, sample_rate):
        collected.append(chunk)
        yield chunk
    if not collected:
        async for chunk in _sapi_tts(text, accent, sample_rate):
            collected.append(chunk)
            yield chunk

    _store_cache(cache_k, collected, text)


# ── Piper (ONNX) ──────────────────────────────────────────────────────────────

def _ensure_piper_voice_files() -> tuple[Optional[str], Optional[str]]:
    """Return (model_path, config_path), downloading the voice once if needed."""
    voices_dir = Path(PIPER_VOICES_DIR)
    try:
        voices_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None, None
    model_path = voices_dir / PIPER_MODEL
    config_path = voices_dir / PIPER_MODEL_JSON

    if model_path.exists() and config_path.exists():
        return str(model_path), str(config_path)

    try:
        import requests
        if not model_path.exists():
            r = requests.get(f"{_PIPER_BASE_URL}/{PIPER_MODEL}", timeout=120)
            r.raise_for_status()
            model_path.write_bytes(r.content)
        if not config_path.exists():
            r = requests.get(f"{_PIPER_BASE_URL}/{PIPER_MODEL_JSON}", timeout=60)
            r.raise_for_status()
            config_path.write_bytes(r.content)
    except Exception:
        # Best-effort: if offline, callers fall back to Edge/SAPI.
        return (str(model_path) if model_path.exists() else None,
                str(config_path) if config_path.exists() else None)

    if model_path.exists():
        return str(model_path), (str(config_path) if config_path.exists() else None)
    return None, None


def _load_piper_voice(model_path: str):
    """Load and cache a PiperVoice (blocking — call via to_thread)."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice
    from piper import PiperVoice  # type: ignore
    _piper_voice = PiperVoice.load(model_path)
    return _piper_voice


def _piper_synthesize_blocking(model_path: str, text: str) -> bytes:
    """Synthesize int16 PCM with Piper (blocking — call via to_thread)."""
    voice = _load_piper_voice(model_path)
    buf = io.BytesIO()
    try:
        for chunk in voice.synthesize_stream_raw(text):
            if chunk:
                buf.write(chunk)
    except Exception:
        return b""
    return buf.getvalue()


def _pcm16_to_f32(pcm16: bytes) -> bytes:
    if len(pcm16) % 2 == 1:
        pcm16 = pcm16[:-1]
    try:
        import numpy as np
        arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return arr.tobytes()
    except Exception:
        i16 = struct.unpack(f"<{len(pcm16)//2}h", pcm16)
        return struct.pack(f"<{len(i16)}f", *[s / 32768.0 for s in i16])


async def _piper_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    model_path, _config = await asyncio.to_thread(_ensure_piper_voice_files)
    if not model_path:
        return
    pcm16 = await asyncio.to_thread(_piper_synthesize_blocking, model_path, text.strip())
    if not pcm16:
        return
    f32 = _pcm16_to_f32(pcm16)
    for chunk in _chunk_bytes(f32, 16384):
        yield chunk


# ── Windows SAPI (offline fallback) ───────────────────────────────────────────

def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _sapi_synthesize_wav_blocking(text: str, accent: str = "american") -> bytes:
    if os.name != "nt" or not str(text).strip():
        return b""
    voice = {
        "british": "Microsoft Hazel Desktop",
        "male": "Microsoft David Desktop",
        "american": "Microsoft Zira Desktop",
    }.get(str(accent).lower(), "Microsoft Zira Desktop")
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"try {{ $s.SelectVoice({_ps_quote(voice)}) }} catch {{}};"
        f"$s.SetOutputToWaveFile({_ps_quote(tmp_name)});"
        f"$s.Speak({_ps_quote(text)});"
        "$s.Dispose();"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        with open(tmp_name, "rb") as fh:
            wav = fh.read()
        return wav if len(wav) > 44 else b""
    except Exception:
        return b""
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


async def _sapi_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    wav = await asyncio.to_thread(_sapi_synthesize_wav_blocking, text, accent)
    if not wav:
        return
    f32 = await asyncio.to_thread(_wav_to_pcm_f32le, wav, sample_rate)
    for chunk in _chunk_bytes(f32, 16384):
        yield chunk


def _wav_to_pcm_f32le(wav_bytes: bytes, target_sample_rate: int = 24000) -> bytes:
    try:
        import miniaudio
        decoded = miniaudio.decode(
            wav_bytes, nchannels=1, sample_rate=target_sample_rate,
            output_format=miniaudio.SampleFormat.FLOAT32,
        )
        return bytes(decoded.samples)
    except Exception:
        return b""


# ── Cartesia ──────────────────────────────────────────────────────────────────

async def _cartesia_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """Cartesia Sonic SSE — streams PCM chunks as they're generated."""
    api_key = os.environ.get("CARTESIA_API_KEY", "")
    voice_id = CARTESIA_VOICES.get(accent) or CARTESIA_VOICES.get("american", "")
    if not api_key or not voice_id:
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            async with client.stream(
                "POST", CARTESIA_URL,
                headers={
                    "X-API-Key": api_key,
                    "Cartesia-Version": "2025-04-16",
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": "sonic-2",
                    "transcript": text,
                    "voice": {"mode": "id", "id": voice_id},
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": sample_rate,
                    },
                },
            ) as response:
                if response.status_code != 200:
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str in ("[DONE]", ""):
                        break
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    if event.get("done"):
                        break
                    if event.get("data"):
                        yield base64.b64decode(event["data"])
    except Exception:
        return


# ── Edge ──────────────────────────────────────────────────────────────────────

async def _edge_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    try:
        import edge_tts

        voice = EDGE_VOICES.get(accent, EDGE_VOICES["american"])
        communicator = edge_tts.Communicate(text, voice)

        mp3_chunks = []
        async for item in communicator.stream():
            if item["type"] == "audio":
                mp3_chunks.append(item["data"])

        if not mp3_chunks:
            return

        chunks = await asyncio.to_thread(
            _mp3_to_pcm_f32le, b"".join(mp3_chunks), sample_rate)
        for chunk in chunks:
            yield chunk

    except ImportError:
        import warnings
        warnings.warn("edge-tts not installed — run: pip install edge-tts", stacklevel=2)
        return
    except Exception:
        return


def _mp3_to_pcm_f32le(mp3_bytes: bytes, target_sample_rate: int = 24000, chunk_ms: int = 200) -> list[bytes]:
    """Decode MP3 → PCM F32LE chunks. miniaudio (no ffmpeg) primary, pydub fallback."""
    try:
        import miniaudio
        decoded = miniaudio.decode(
            mp3_bytes, nchannels=1, sample_rate=target_sample_rate,
            output_format=miniaudio.SampleFormat.FLOAT32,
        )
        f32 = bytes(decoded.samples)
        size = (target_sample_rate * chunk_ms // 1000) * 4
        return [f32[i:i + size] for i in range(0, len(f32), size)]
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes)).set_frame_rate(target_sample_rate).set_channels(1)
        raw = audio.raw_data
        i16 = struct.unpack(f"<{len(raw)//2}h", raw)
        f32 = struct.pack(f"<{len(i16)}f", *[s / 32768.0 for s in i16])
        size = (target_sample_rate * chunk_ms // 1000) * 4
        return [f32[i:i + size] for i in range(0, len(f32), size)]
    except Exception:
        return []
