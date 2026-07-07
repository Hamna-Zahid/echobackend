"""
TTS service — Microsoft Edge Neural TTS via edge-tts (FREE, no API key).

Returns raw PCM F32LE chunks at 24 kHz (what the mobile client plays). Good
neural voice quality (Azure voices under the hood), zero cost.

DEV_TTS modes:
  edge    → Edge Neural TTS (default, free)
  silent  → 0.3 s of silence per call (AI-logic testing only, fastest)
"""

import os
import io
import struct
from typing import AsyncGenerator

EDGE_VOICES = {
    "american": "en-US-AriaNeural",
    "british":  "en-GB-SoniaNeural",
    "male":     "en-US-GuyNeural",
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

    async for chunk in _edge_tts(text, accent, sample_rate):
        yield chunk


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
