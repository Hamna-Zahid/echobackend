"""
TTS service — Kokoro-82M primary (local, free), edge-tts fallback (free).

Stack:
  kokoro   → 82M param neural TTS, runs on CPU, ~80-120ms first chunk
             American voices: af_bella, af_sarah, am_adam, am_michael
             British voices:  bf_emma, bf_isabella, bm_george, bm_lewis
             Install: pip install kokoro soundfile

  edge-tts → Microsoft Azure Neural voices, free, no key
             Falls back automatically if kokoro is not installed

  silent   → Zero-latency silence for AI logic testing (DEV_TTS=silent)

Set in backend/.env:
  DEV_TTS=kokoro   ← local Kokoro (recommended — free, fast, expressive)
  DEV_TTS=edge     ← Microsoft Edge TTS (free fallback)
  DEV_TTS=silent   ← pure AI testing (no audio)
"""

import asyncio
import io
import os
import struct
from typing import AsyncGenerator

# ── Config ────────────────────────────────────────────────────────────────────

_TTS_MODE = os.environ.get("DEV_TTS", "kokoro").lower()

# Kokoro voice mapping
KOKORO_VOICES = {
    "american": os.environ.get("KOKORO_VOICE_AMERICAN", "af_bella"),
    "british":  os.environ.get("KOKORO_VOICE_BRITISH", "bf_emma"),
}

# Edge TTS fallback voices
EDGE_VOICES = {
    "american": "en-US-AriaNeural",
    "british":  "en-GB-SoniaNeural",
}

# Kokoro pipeline (singleton — load once, reuse)
_kokoro_pipeline = None
_kokoro_lock = asyncio.Lock()


async def _get_kokoro_pipeline(lang_code: str = "a"):
    global _kokoro_pipeline
    async with _kokoro_lock:
        if _kokoro_pipeline is None:
            from kokoro import KPipeline
            # KPipeline loads ~100MB model weights on first call
            _kokoro_pipeline = KPipeline(lang_code=lang_code)
    return _kokoro_pipeline


# ── Public interface ──────────────────────────────────────────────────────────

async def stream_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """
    Stream PCM F32LE audio chunks for the given text.
    Routing: kokoro → edge-tts → silence.
    """
    if not text or not text.strip():
        return

    mode = _TTS_MODE

    if mode == "silent":
        num_samples = int(sample_rate * 0.3)
        yield bytes(num_samples * 4)
        return

    if mode == "edge":
        async for chunk in _edge_tts(text, accent, sample_rate):
            yield chunk
        return

    # Default: Kokoro (local, free)
    async for chunk in _kokoro_tts(text, accent, sample_rate):
        yield chunk


# ── Kokoro-82M (local, free) ──────────────────────────────────────────────────

async def _kokoro_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """
    Kokoro-82M local neural TTS.
    Runs on CPU with ~80-120ms first chunk latency.
    Yields PCM F32LE bytes in sentence-length chunks for streaming playback.
    """
    try:
        import numpy as np

        lang_code = "a" if accent == "american" else "b"
        voice = KOKORO_VOICES.get(accent, KOKORO_VOICES["american"])

        pipeline = await _get_kokoro_pipeline(lang_code)

        # Run CPU-bound synthesis in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _synthesize():
            chunks = []
            generator = pipeline(text, voice=voice, speed=1.0)
            for _graphemes, _phonemes, audio in generator:
                if audio is not None and len(audio) > 0:
                    # audio is numpy float32 normalized to [-1, 1]
                    # Convert to F32LE bytes (what the client expects)
                    chunks.append(audio.astype(np.float32).tobytes())
            return chunks

        audio_chunks = await loop.run_in_executor(None, _synthesize)

        for chunk in audio_chunks:
            if chunk:
                yield chunk

    except ImportError:
        import warnings
        warnings.warn(
            "kokoro not installed. Run: pip install kokoro soundfile\n"
            "Falling back to edge-tts.",
            stacklevel=2,
        )
        async for chunk in _edge_tts(text, accent, sample_rate):
            yield chunk

    except Exception as e:
        import warnings
        warnings.warn(f"Kokoro TTS error: {e} — falling back to edge-tts")
        async for chunk in _edge_tts(text, accent, sample_rate):
            yield chunk


# ── Edge TTS (free fallback) ──────────────────────────────────────────────────

async def _edge_tts(
    text: str,
    accent: str = "american",
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    """Microsoft Edge TTS — free, no API key, good quality neural voices."""
    try:
        import edge_tts

        voice = EDGE_VOICES.get(accent, EDGE_VOICES["american"])
        communicator = edge_tts.Communicate(text, voice)

        # edge-tts hits an unofficial Microsoft endpoint that can occasionally stall.
        # Cap the whole fetch so one slow turn can't hang for 25s+ — fail fast and let
        # the caller move on (the session stays usable, text is already shown).
        async def _collect():
            chunks = []
            async for item in communicator.stream():
                if item["type"] == "audio":
                    chunks.append(item["data"])
            return chunks

        mp3_chunks = await asyncio.wait_for(_collect(), timeout=12.0)

        if not mp3_chunks:
            yield b""
            return

        mp3_bytes = b"".join(mp3_chunks)
        for chunk in _mp3_to_pcm_f32le(mp3_bytes, sample_rate):
            yield chunk

    except ImportError:
        import warnings
        warnings.warn("edge-tts not installed. Run: pip install edge-tts")
        yield bytes(int(sample_rate * 0.5) * 4)

    except Exception:
        yield bytes(int(sample_rate * 0.3) * 4)


def _mp3_to_pcm_f32le(mp3_bytes: bytes, target_sample_rate: int = 24000) -> list[bytes]:
    """Convert MP3 bytes to PCM F32LE chunks (~200ms each)."""
    chunk_size_ms = 200

    try:
        import miniaudio
        decoded = miniaudio.decode(
            mp3_bytes,
            nchannels=1,
            sample_rate=target_sample_rate,
            output_format=miniaudio.SampleFormat.FLOAT32,
        )
        f32_bytes = bytes(decoded.samples)
        chunk_bytes = (target_sample_rate * chunk_size_ms // 1000) * 4
        return [f32_bytes[i:i + chunk_bytes] for i in range(0, len(f32_bytes), chunk_bytes)]
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
        raw_pcm = audio.raw_data
        samples_i16 = struct.unpack(f"<{len(raw_pcm) // 2}h", raw_pcm)
        f32_bytes = struct.pack(f"<{len(samples_i16)}f", *[s / 32768.0 for s in samples_i16])
        chunk_bytes = (target_sample_rate * chunk_size_ms // 1000) * 4
        return [f32_bytes[i:i + chunk_bytes] for i in range(0, len(f32_bytes), chunk_bytes)]
    except Exception:
        return [bytes(target_sample_rate * 4)]
