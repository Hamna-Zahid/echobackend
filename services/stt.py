"""
STT service — Groq Whisper Large-v3 (FREE tier: ~14,400 requests/day, no card).

Returns the transcript text plus a few delivery signals we need for the feedback
engine: total spoken duration (for words-per-minute) and Whisper's segment-level
confidence (a rough pronunciation/clarity proxy).

Everything here runs on Groq's free tier — there is no GPU to host and no per-use
charge while you build.
"""

import io
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import AsyncGroq

_client: AsyncGroq | None = None

WHISPER_MODEL = "whisper-large-v3"

# Common English filler words/phrases we score delivery against.
FILLER_PATTERNS = [
    r"\bum\b", r"\buh\b", r"\ber\b", r"\bah\b", r"\bhmm\b",
    r"\blike\b", r"\byou know\b", r"\bsort of\b", r"\bkind of\b",
    r"\bbasically\b", r"\bactually\b", r"\bi mean\b", r"\bso\b",
]


@dataclass
class TranscriptResult:
    text: str
    duration_sec: float = 0.0           # spoken length of this utterance
    avg_confidence: Optional[float] = None
    filler_hits: dict = field(default_factory=dict)  # {"um": 3, "like": 2, ...}


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """Plain transcript string (used for the live conversation turn)."""
    result = await transcribe_audio_full(audio_bytes, language)
    return result.text


async def transcribe_audio_full(audio_bytes: bytes, language: str = "en") -> TranscriptResult:
    """
    Full transcription with delivery metadata (duration + filler counts).
    Used so the feedback engine can compute accurate WPM and filler stats from
    real numbers rather than letting the LLM guess.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return TranscriptResult(text="")

    client = get_groq_client()
    audio_file = io.BytesIO(audio_bytes)
    # The filename extension tells Groq which decoder to use — sniff the real
    # container from magic bytes (Android sends m4a/AAC, iOS wav, older builds webm).
    audio_file.name = f"audio.{_sniff_container(audio_bytes)}"

    try:
        transcription = await client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL,
            language=language,
            response_format="verbose_json",
        )
        text = (getattr(transcription, "text", "") or "").strip()
        duration = float(getattr(transcription, "duration", 0.0) or 0.0)
        avg_conf = _avg_confidence(transcription)
        return TranscriptResult(
            text=text,
            duration_sec=duration,
            avg_confidence=avg_conf,
            filler_hits=count_fillers(text),
        )
    except Exception:
        # Resilient fallback: plain text, no metadata.
        try:
            audio_file.seek(0)
            transcription = await client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                language=language,
                response_format="text",
            )
            text = (transcription if isinstance(transcription, str) else getattr(transcription, "text", "")).strip()
            return TranscriptResult(text=text, filler_hits=count_fillers(text))
        except Exception:
            return TranscriptResult(text="")


def _sniff_container(audio: bytes) -> str:
    """Detect the audio container from magic bytes so Groq picks the right decoder."""
    if len(audio) < 12:
        return "webm"
    if audio[:4] == b"RIFF":
        return "wav"
    if audio[4:8] == b"ftyp":
        return "m4a"
    if audio[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if audio[:4] == b"OggS":
        return "ogg"
    if audio[:3] == b"ID3" or (audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0):
        return "mp3"
    return "webm"


def count_fillers(text: str) -> dict:
    """Count filler words/phrases in a piece of transcript."""
    low = text.lower()
    hits: dict = {}
    for pat in FILLER_PATTERNS:
        n = len(re.findall(pat, low))
        if n:
            label = pat.replace(r"\b", "").replace("\\", "")
            hits[label] = hits.get(label, 0) + n
    return hits


def _avg_confidence(transcription) -> Optional[float]:
    """Mean avg_logprob across segments, mapped to a 0–100 clarity proxy."""
    try:
        segments = getattr(transcription, "segments", None)
        if not segments:
            return None
        logprobs = [getattr(s, "avg_logprob", None) for s in segments]
        logprobs = [lp for lp in logprobs if lp is not None]
        if not logprobs:
            return None
        avg = sum(logprobs) / len(logprobs)
        return round(max(0.0, min(100.0, (1.0 + avg) * 100)), 1)
    except Exception:
        return None
