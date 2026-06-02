"""
STT service — Groq Whisper Large-v3.

Returns both the transcript text AND pronunciation metadata extracted from
Whisper's segment-level confidence scores (avg_logprob).

avg_logprob interpretation:
  > -0.2   → confident recognition → likely clear pronunciation
  -0.2 to -0.5 → moderate confidence → possible pronunciation issue
  < -0.5   → low confidence → likely pronunciation difficulty

This is a proxy for pronunciation quality, not a dedicated phoneme model.
For production-grade pronunciation scoring, wire in Azure Cognitive Services
Speech SDK (see services/pronunciation.py).
"""

import io
import os
from dataclasses import dataclass
from typing import Optional

from groq import AsyncGroq

_client: AsyncGroq | None = None

# Threshold below which a segment is flagged as potentially mispronounced
PRONUNCIATION_FLAG_THRESHOLD = -0.4


@dataclass
class TranscriptResult:
    text: str
    pronunciation_score: Optional[float]   # 0–100 proxy score, None if unavailable
    flagged_segments: list[str]             # words/phrases with low confidence
    avg_confidence: Optional[float]         # raw avg_logprob averaged across segments


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def transcribe_audio(
    audio_bytes: bytes,
    language: str = "en",
    include_pronunciation_hints: bool = True,
) -> str:
    """
    Transcribe audio using Groq Whisper Large-v3.
    Returns plain text transcript string.
    Target latency: < 200ms for 5–10s utterances.
    """
    result = await transcribe_audio_full(audio_bytes, language, include_pronunciation_hints)
    return result.text


async def transcribe_audio_full(
    audio_bytes: bytes,
    language: str = "en",
    include_pronunciation_hints: bool = True,
) -> TranscriptResult:
    """
    Full transcription with pronunciation metadata.
    Used by diagnostic mode and pronunciation scoring pipeline.
    """
    client = get_groq_client()

    if not audio_bytes or len(audio_bytes) < 100:
        return TranscriptResult(text="", pronunciation_score=None, flagged_segments=[], avg_confidence=None)

    audio_file = io.BytesIO(audio_bytes)
    # Set filename so Groq infers correct codec
    # Android sends webm/opus, iOS sends wav/PCM — Whisper handles both
    audio_file.name = "audio.webm"

    try:
        if include_pronunciation_hints:
            # verbose_json gives us segment-level confidence (avg_logprob)
            transcription = await client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=language,
                response_format="verbose_json",
            )
            text = transcription.text.strip() if hasattr(transcription, "text") else ""
            pronunciation_result = _extract_pronunciation_hints(transcription)
            return TranscriptResult(
                text=text,
                pronunciation_score=pronunciation_result["score"],
                flagged_segments=pronunciation_result["flagged"],
                avg_confidence=pronunciation_result["avg_logprob"],
            )
        else:
            transcription = await client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=language,
                response_format="text",
            )
            text = transcription if isinstance(transcription, str) else transcription.text
            return TranscriptResult(
                text=text.strip(),
                pronunciation_score=None,
                flagged_segments=[],
                avg_confidence=None,
            )

    except Exception as e:
        # Retry once with plain text format (more resilient)
        try:
            audio_file.seek(0)
            transcription = await client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=language,
                response_format="text",
            )
            text = transcription if isinstance(transcription, str) else transcription.text
            return TranscriptResult(
                text=text.strip() if text else "",
                pronunciation_score=None,
                flagged_segments=[],
                avg_confidence=None,
            )
        except Exception:
            return TranscriptResult(text="", pronunciation_score=None, flagged_segments=[], avg_confidence=None)


def _extract_pronunciation_hints(transcription) -> dict:
    """
    Extract pronunciation confidence hints from Whisper verbose_json segments.

    avg_logprob per segment is the mean log-probability of all tokens in that segment.
    We use it as a proxy for pronunciation clarity.

    Returns:
      score: 0–100 (100 = perfectly clear, 0 = unintelligible)
      flagged: list of segment texts with low confidence
      avg_logprob: raw average across all segments
    """
    try:
        segments = getattr(transcription, "segments", None)
        if not segments:
            return {"score": None, "flagged": [], "avg_logprob": None}

        logprobs = []
        flagged = []

        for seg in segments:
            logprob = getattr(seg, "avg_logprob", None)
            if logprob is not None:
                logprobs.append(logprob)
                if logprob < PRONUNCIATION_FLAG_THRESHOLD:
                    text = getattr(seg, "text", "").strip()
                    if text:
                        flagged.append(text)

        if not logprobs:
            return {"score": None, "flagged": [], "avg_logprob": None}

        avg_logprob = sum(logprobs) / len(logprobs)

        # Map avg_logprob to 0–100 score
        # logprob range: 0 (perfect) to -inf (unintelligible)
        # Practical range: 0 to -1.0 for normal speech
        # We map: 0 → 100, -0.5 → 50, -1.0 → 0 (clamp below 0)
        score = max(0.0, min(100.0, (1.0 + avg_logprob) * 100))

        return {
            "score": round(score, 1),
            "flagged": flagged[:3],  # Top 3 flagged segments max
            "avg_logprob": round(avg_logprob, 4),
        }

    except Exception:
        return {"score": None, "flagged": [], "avg_logprob": None}
