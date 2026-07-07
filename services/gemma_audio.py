"""
Gemma 4 audio-native delivery analyzer — FREE via Google AI Studio.

Gemma 4 (12B unified, released April 2026) natively ingests AUDIO and returns
text. Unlike the Whisper transcript, it hears *how* the user spoke: hesitation,
tone, confidence, pronunciation. We use it as a sidecar quality layer on top of
the Groq cascade — never as the primary path:

  per user turn:  audio ──► Gemma 4 12B ──► delivery JSON
                   ├─ live proactive coach tip (throttled, correction_card UI)
                   └─ collected → injected into the final 70B scorecard

Design constraints honoured:
  * Gemma 4 audio clips max out at 30 s → longer answers are skipped
    (the Whisper transcript path has no such limit and still covers them).
  * Fully degradable: no GEMINI_API_KEY, network error, quota hit → returns
    None and the session behaves exactly as before.
  * Free tier only. REST via httpx — no new SDK dependency.
"""

import base64
import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("echo.gemma")

GEMMA_MODEL = os.environ.get("GEMMA_AUDIO_MODEL", "gemma-4-12b-it")
_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

# Gemma 4 hard limit: 30 s of audio per clip. We leave headroom.
MAX_AUDIO_SECONDS = 29.0
# Don't bother analysing tiny clips ("yes", "okay").
MIN_AUDIO_SECONDS = 3.0

_ANALYSIS_PROMPT = """You are an expert speaking coach listening to one answer from a practice interview/speech session.

Listen to HOW the person speaks, not just the words. Return ONLY a JSON object:

{
  "confidence": <int 1-10, how confident they SOUND>,
  "tone": "<2-4 words, e.g. 'flat and hesitant', 'warm, engaged'>",
  "hesitancy": "low" | "medium" | "high",
  "pronunciation_flags": ["<specific word/sound they struggled with>", ...max 2],
  "delivery_tip": "<ONE short actionable tip about their DELIVERY (voice, pace, energy, hesitation) — or empty string if delivery was fine>"
}

Be specific to what you heard. No markdown, JSON only."""


def _sniff_mime(audio: bytes) -> str:
    """Detect container from magic bytes (Android sends webm/opus, iOS wav)."""
    if audio[:4] == b"RIFF":
        return "audio/wav"
    if audio[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"
    if audio[:3] == b"ID3" or (len(audio) > 1 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0):
        return "audio/mp3"
    if audio[:4] == b"OggS":
        return "audio/ogg"
    return "audio/webm"


def is_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


async def analyze_delivery(audio_bytes: bytes, duration_sec: float) -> Optional[dict]:
    """
    Send one utterance's audio to Gemma 4 and get a delivery judgment.
    Returns the parsed dict, or None whenever anything prevents analysis
    (no key, clip too long/short, network/quota error, bad response).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not audio_bytes:
        return None
    if not (MIN_AUDIO_SECONDS <= duration_sec <= MAX_AUDIO_SECONDS):
        return None

    payload = {
        "contents": [{
            "parts": [
                {"text": _ANALYSIS_PROMPT},
                {"inline_data": {
                    "mime_type": _sniff_mime(audio_bytes),
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                }},
            ],
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 250},
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0)) as client:
            res = await client.post(
                _API_URL.format(model=GEMMA_MODEL),
                params={"key": api_key},
                json=payload,
            )
        if res.status_code != 200:
            logger.debug("Gemma audio HTTP %s: %s", res.status_code, res.text[:200])
            return None

        data = res.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        result = json.loads(text[start:end + 1])
        if not isinstance(result, dict) or "confidence" not in result:
            return None
        return result
    except Exception as e:
        logger.debug("Gemma audio analysis failed: %s", e)
        return None


def format_delivery_notes(notes: list[dict]) -> str:
    """Render collected per-turn delivery analyses for the scorecard prompt."""
    if not notes:
        return ""
    lines = []
    for i, n in enumerate(notes, 1):
        flags = ", ".join(n.get("pronunciation_flags") or []) or "none"
        lines.append(
            f"  answer {i}: confidence={n.get('confidence')}/10, tone='{n.get('tone', '')}', "
            f"hesitancy={n.get('hesitancy', '?')}, pronunciation issues: {flags}"
        )
    return (
        "AUDIO DELIVERY ANALYSIS (an audio-native model listened to the actual "
        "recordings — use this for the clarity/confidence scores and delivery "
        "feedback; it hears what the transcript cannot):\n" + "\n".join(lines)
    )
