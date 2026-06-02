"""
Pronunciation scoring service.

Tier 1 (current): Whisper confidence proxy
  — Available now, zero extra cost, ~70% accuracy
  — Uses avg_logprob from verbose_json segments
  — Good enough for initial IELTS band estimation

Tier 2 (production): Azure Cognitive Services Speech SDK
  — Phoneme-level accuracy assessment per word
  — Returns accuracy, fluency, completeness, and prosody scores
  — Maps directly to IELTS Pronunciation band descriptor
  — Requires AZURE_SPEECH_KEY + AZURE_SPEECH_REGION env vars
  — Install: pip install azure-cognitiveservices-speech

Tier 3 (fine-tuned): Custom model
  — Fine-tune Wav2Vec2 on South Asian English phoneme errors
  — Provides granular feedback: "you said /p/ but it sounds like /b/"
  — Requires ~500 hours labeled South Asian English data
  — Not in scope for v1 launch

The service auto-detects which tier is available and uses the best one.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PronunciationResult:
    overall_score: float          # 0–100 (maps to IELTS 0–9 band)
    ielts_band: float             # 0–9
    accuracy_score: Optional[float]   # phoneme accuracy (Tier 2+)
    fluency_score: Optional[float]    # speaking rate, pausing (Tier 2+)
    prosody_score: Optional[float]    # rhythm, stress, intonation (Tier 2+)
    flagged_words: list[str]          # words to work on
    feedback: str                     # one-sentence actionable tip
    tier_used: str                    # "whisper_proxy" | "azure" | "custom"


# ── Whisper proxy scorer (Tier 1) ─────────────────────────────────────────────

def score_from_whisper(
    pronunciation_score: Optional[float],
    flagged_segments: list[str],
    transcript: str,
) -> PronunciationResult:
    """
    Convert Whisper confidence proxy to a PronunciationResult.
    Used when Azure is not configured.
    """
    if pronunciation_score is None:
        # No confidence data — return neutral score
        return PronunciationResult(
            overall_score=70.0,
            ielts_band=6.0,
            accuracy_score=None,
            fluency_score=None,
            prosody_score=None,
            flagged_words=[],
            feedback="Keep speaking clearly and at a natural pace.",
            tier_used="whisper_proxy",
        )

    ielts_band = _score_to_ielts_band(pronunciation_score)
    feedback = _generate_proxy_feedback(pronunciation_score, flagged_segments)

    return PronunciationResult(
        overall_score=pronunciation_score,
        ielts_band=ielts_band,
        accuracy_score=None,
        fluency_score=None,
        prosody_score=None,
        flagged_words=flagged_segments,
        feedback=feedback,
        tier_used="whisper_proxy",
    )


# ── Azure Speech scorer (Tier 2) ─────────────────────────────────────────────

def _azure_available() -> bool:
    return bool(os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION"))


async def score_pronunciation_azure(
    audio_bytes: bytes,
    reference_text: str,
) -> Optional[PronunciationResult]:
    """
    Full pronunciation assessment via Azure Cognitive Services.
    Returns None if Azure is not configured or call fails.

    Requires:
      pip install azure-cognitiveservices-speech
      AZURE_SPEECH_KEY=<your-key>
      AZURE_SPEECH_REGION=<e.g. eastus>
    """
    if not _azure_available():
        return None

    try:
        import azure.cognitiveservices.speech as speechsdk
        import asyncio

        speech_key = os.environ["AZURE_SPEECH_KEY"]
        speech_region = os.environ["AZURE_SPEECH_REGION"]

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = "en-US"

        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True,
        )

        # Write audio bytes to a stream
        audio_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        audio_stream.write(audio_bytes)
        audio_stream.close()

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        pronunciation_config.apply_to(recognizer)

        # Run recognition in executor to avoid blocking async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, recognizer.recognize_once)

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            assessment = speechsdk.PronunciationAssessmentResult(result)

            overall = assessment.pronunciation_score
            accuracy = assessment.accuracy_score
            fluency = assessment.fluency_score
            prosody = getattr(assessment, "prosody_score", None)

            # Find mispronounced words
            flagged_words = []
            for word in assessment.words:
                if word.accuracy_score < 60:
                    flagged_words.append(word.word)

            ielts_band = _score_to_ielts_band(overall)
            feedback = _generate_azure_feedback(accuracy, fluency, prosody, flagged_words)

            return PronunciationResult(
                overall_score=overall,
                ielts_band=ielts_band,
                accuracy_score=accuracy,
                fluency_score=fluency,
                prosody_score=prosody,
                flagged_words=flagged_words[:5],
                feedback=feedback,
                tier_used="azure",
            )

    except Exception:
        return None

    return None


# ── Unified scoring entry point ───────────────────────────────────────────────

async def score_pronunciation(
    audio_bytes: bytes,
    transcript: str,
    pronunciation_score_proxy: Optional[float] = None,
    flagged_segments: Optional[list] = None,
) -> PronunciationResult:
    """
    Score pronunciation using the best available tier.
    Called from ws.py after each diagnostic turn.
    """
    # Try Azure first
    if _azure_available() and audio_bytes and transcript:
        azure_result = await score_pronunciation_azure(audio_bytes, transcript)
        if azure_result:
            return azure_result

    # Fall back to Whisper proxy
    return score_from_whisper(
        pronunciation_score=pronunciation_score_proxy,
        flagged_segments=flagged_segments or [],
        transcript=transcript,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_to_ielts_band(score: float) -> float:
    """Map 0–100 pronunciation score to IELTS 0–9 band."""
    # IELTS bands are coarse — map linearly then round to nearest 0.5
    raw = score / 100.0 * 9.0
    # Round to nearest 0.5
    return round(raw * 2) / 2


def _generate_proxy_feedback(score: float, flagged: list[str]) -> str:
    """Generate actionable feedback from Whisper proxy score."""
    if score >= 85:
        return "Pronunciation is clear and easy to follow — keep this up."
    elif score >= 70:
        if flagged:
            return f"Generally clear, but work on articulating phrases like: {', '.join(flagged[:2])}."
        return "Generally clear — focus on word stress and linking sounds between words."
    elif score >= 55:
        if flagged:
            return f"Some parts were harder to follow — especially: {', '.join(flagged[:2])}. Slow down and over-articulate those."
        return "Pronunciation needs attention — try speaking slightly slower and opening your mouth more on vowels."
    else:
        return "Intelligibility is reduced — practice reading aloud slowly for 10 minutes daily, focusing on clear consonant endings."


def _generate_azure_feedback(
    accuracy: float,
    fluency: float,
    prosody: Optional[float],
    flagged_words: list[str],
) -> str:
    """Generate specific feedback from Azure assessment scores."""
    if accuracy < 60:
        if flagged_words:
            return f"Focus on word accuracy — practice these words slowly: {', '.join(flagged_words[:3])}."
        return "Phoneme accuracy is the priority — try shadowing native speakers sentence by sentence."
    elif fluency < 65:
        return "Work on speaking fluency — reduce pausing within phrases and connect words more naturally."
    elif prosody is not None and prosody < 60:
        return "Work on natural rhythm — English uses stress-timed rhythm; stressed syllables should be louder and longer."
    elif flagged_words:
        return f"Good overall, but polish these specific words: {', '.join(flagged_words[:3])}."
    else:
        return "Pronunciation is strong — focus on maintaining this under speaking pressure."
