"""
TTS service — streaming voice synthesis.

DEV_TTS modes:
  edge     → Microsoft Edge Neural TTS (default — FREE, no key, good quality)
  cartesia → Cartesia Sonic streaming (pro voice quality, true progressive
             streaming; needs CARTESIA_API_KEY + CARTESIA_VOICE_ID_* env;
             free monthly allowance, then paid). Auto-falls back to edge on
             any failure — flipping this env var can never break sessions.
  silent   → 0.3 s of silence per call (AI-logic testing only, fastest)

Always yields raw PCM F32LE chunks at 24 kHz (what the mobile client plays).
"""

import base64
import json
import os
import io
import struct
from typing import AsyncGenerator

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

_TTS_MODE = os.environ.get("DEV_TTS", "edge").lower()


async def stream_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """Stream PCM F32LE audio chunks for `text`. Never buffers the full reply."""
    if not text or not text.strip():
        return

    if _TTS_MODE == "silent":
        yield bytes(int(sample_rate * 0.3) * 4)
        return

    if _TTS_MODE == "cartesia":
        produced = False
        async for chunk in _cartesia_tts(text, accent, sample_rate):
            produced = True
            yield chunk
        if produced:
            return
        # Cartesia unavailable/failed — fall through to edge so the session
        # keeps speaking no matter what.

    async for chunk in _edge_tts(text, accent, sample_rate):
        yield chunk


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
            yield b""
            return

        for chunk in _mp3_to_pcm_f32le(b"".join(mp3_chunks), sample_rate):
            yield chunk

    except ImportError:
        import warnings
        warnings.warn("edge-tts not installed — run: pip install edge-tts", stacklevel=2)
        yield bytes(int(sample_rate * 0.5) * 4)
    except Exception:
        yield bytes(int(sample_rate * 0.3) * 4)


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
        return [bytes(target_sample_rate * 4)]
