"""
ASR Pipeline — Two-stage Automatic Speech Recognition.

Architecture (inspired by the Interspeech 2025 Speech Accessibility Project
Challenge: 17.82% WER baseline -> 8.11% WER through a cascaded pipeline):

  Stage 1 (detector)   : Whisper tiny on Groq — fast speech-presence / gating.
  Stage 2 (recognizer) : Whisper large-v3 on Groq — accurate final transcript.
  Post-processing      : Coach vocabulary awareness + light normalization.
  Quality              : WER estimate + quality tier from segment confidence.

Free-tier compatible: both stages run on Groq's hosted Whisper, no GPU.
"""

import io
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Set

from groq import AsyncGroq

from services.vad_service import ImprovedVAD, detect_speech_simple, DEFAULT_CONFIG
from services.stt import (
    count_fillers,
    _avg_confidence,
    _sniff_container,
    _wrap_pcm_as_wav,
    COACH_VOCABULARY_TERMS,
)
from services.domain_adaptation import get_coach_adapter, rerank_coach_transcript

logger = logging.getLogger("echo.asr_pipeline")

# ── Model selection ──────────────────────────────────────────────────────────
WHISPER_DETECTOR_MODEL = "whisper-large-v3-turbo"
WHISPER_RECOGNIZER_MODEL = "whisper-large-v3"

# Backwards-compatible alias (ws.py imports this name).
COACH_VOCABULARY = [
    "listening", "thinking", "speaking",
    "interrupt", "session_end", "feedback",
    "goal", "progress", "improvement",
    "clarity", "confidence", "pace",
    "coach", "coaching", "improvements",
]

_AUDIO_CONTAINERS = ("wav", "m4a", "webm", "ogg", "mp3")


# ── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class ASRResult:
    """Structured result from the ASR pipeline."""
    text: str
    duration_sec: float = 0.0
    language: str = "en"
    confidence: Optional[float] = None          # 0-100 clarity proxy from avg_logprob
    filler_hits: dict = field(default_factory=dict)
    stt_stage: str = "two_stage"                # detector | recognizer | two_stage | vad_silence | error
    detector_time_ms: Optional[int] = None
    recognizer_time_ms: Optional[int] = None
    vad_used: bool = False
    vocab_boosted: Optional[Set[str]] = None
    wers_estimate: Optional[float] = None       # Estimated WER proxy (0-30)
    quality_tier: str = "fair"                  # excellent | good | fair | poor
    coach_vocab_detected: bool = False


# ── Module-level Groq clients (reused across sessions) ───────────────────────
_detector_client: Optional[AsyncGroq] = None
_recognizer_client: Optional[AsyncGroq] = None


def _get_detector_client() -> AsyncGroq:
    global _detector_client
    if _detector_client is None:
        _detector_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _detector_client


def _get_recognizer_client() -> AsyncGroq:
    global _recognizer_client
    if _recognizer_client is None:
        _recognizer_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _recognizer_client


def _prepare_audio(audio_bytes: bytes):
    """Return (bytes, container) guaranteed to be a Groq-friendly container."""
    container = _sniff_container(audio_bytes)
    if container not in ("wav", "m4a", "webm", "ogg", "mp3"):
        return _wrap_pcm_as_wav(audio_bytes), "wav"
    return audio_bytes, container


# ── ASR Pipeline Class ────────────────────────────────────────────────────────

class ASRPipeline:
    """
    Two-stage ASR pipeline with VAD gating and coach language post-processing.

    Pipeline flow:
    1. VAD gate  — skip if the turn is silence.
    2. Stage 1   — Whisper tiny (fast detector).
    3. Stage 2   — Whisper large-v3 (accurate recognizer, authoritative).
    4. Reranking — coach vocabulary boost / normalization.
    5. Metrics   — WER estimate + quality tier.
    """

    def __init__(self,
                 detector_size: str = "tiny",
                 recognizer_size: str = "large-v3",
                 vad_config: str = DEFAULT_CONFIG):
        self.detector_size = detector_size
        self.recognizer_size = recognizer_size
        # VAD is also applied in the WS handler; keep an instance for
        # pipeline-level use without duplicating per-turn state.
        self.vad = ImprovedVAD(config_name=vad_config, sample_rate=16000)
        self.coach_vocabulary = COACH_VOCABULARY_TERMS

        logger.info("ASRPipeline initialized: detector=%s, recognizer=%s, vad=%s",
                    detector_size, recognizer_size, vad_config)

    # ── Core Pipeline ──────────────────────────────────────────────────────

    async def transcribe(self, audio_bytes: bytes,
                         use_vad: bool = True) -> ASRResult:
        """Execute the two-stage ASR pipeline and return an ASRResult."""
        import time

        if not audio_bytes:
            return ASRResult(text="", duration_sec=0.0, stt_stage="error",
                             quality_tier="poor", wers_estimate=None)

        # ── Step 1: VAD gating ─────────────────────────────────────────
        if use_vad:
            vad_result = detect_speech_simple(audio_bytes, DEFAULT_CONFIG, 16000)
            if vad_result is not None and not vad_result.is_speech:
                logger.debug("VAD: silence (dBFS=%.2f)", vad_result.dbfs)
                return ASRResult(text="", duration_sec=0.0, stt_stage="vad_silence",
                                 quality_tier="fair", vad_used=True)

        send_bytes, container = _prepare_audio(audio_bytes)

        # ── Step 2: Stage 1 — fast detector ─────────────────────────────
        detector_text = ""
        detector_conf: Optional[float] = None
        t0 = time.time()
        try:
            f = io.BytesIO(send_bytes)
            f.name = f"audio.{container}"
            transcription = await _get_detector_client().audio.transcriptions.create(
                file=f,
                model=WHISPER_DETECTOR_MODEL,
                language="en",
                response_format="verbose_json",
            )
            detector_text = (getattr(transcription, "text", "") or "").strip()
            detector_conf = _avg_confidence(transcription) or 0.0
        except Exception as e:
            logger.warning("Stage 1 (detector) failed: %s", e)
            detector_conf = 0.0
        detector_time_ms = int((time.time() - t0) * 1000)

        # ── Step 2b: Stage 2 — accurate recognizer (authoritative) ───────
        t1 = time.time()
        recognizer_text = ""
        recognizer_duration = 0.0
        recognizer_conf: Optional[float] = None
        recognizer_fillers: dict = {}
        detected_language = "en"
        try:
            f2 = io.BytesIO(send_bytes)
            f2.name = f"audio.{container}"
            transcription = await _get_recognizer_client().audio.transcriptions.create(
                file=f2,
                model=WHISPER_RECOGNIZER_MODEL,
                language="en",
                response_format="verbose_json",
            )
            recognizer_text = (getattr(transcription, "text", "") or "").strip()
            recognizer_duration = float(getattr(transcription, "duration", 0.0) or 0.0)
            recognizer_conf = _avg_confidence(transcription)
            detected_language = getattr(transcription, "language", "en") or "en"
            recognizer_fillers = count_fillers(recognizer_text) if recognizer_text else {}
        except Exception as e:
            logger.error("Stage 2 (recognizer) failed: %s", e)
        recognizer_time_ms = int((time.time() - t1) * 1000)

        # ── Step 3: Coach vocabulary awareness ───────────────────────────
        low = recognizer_text.lower()
        vocab_boosted: Set[str] = {
            term for term in self.coach_vocabulary
            if low and re.search(rf"\b{re.escape(term)}\b", low)
        }
        coach_vocab_detected = bool(vocab_boosted)

        # ── Step 3b: Language-model reranking / normalization ────────────
        try:
            final_text, rerank_confidence = self._rerank_with_language_model(recognizer_text)
        except Exception:
            final_text, rerank_confidence = recognizer_text, (recognizer_conf or 0.0)

        # ── Step 4: Quality metrics ──────────────────────────────────────
        wers_estimate = self._estimate_wer(rerank_confidence, detector_conf or 0.0)
        quality_tier = self._classify_quality(wers_estimate, recognizer_conf)

        return ASRResult(
            text=final_text,
            duration_sec=recognizer_duration,
            language=detected_language,
            confidence=recognizer_conf,
            filler_hits=recognizer_fillers,
            stt_stage="two_stage" if (detector_text or recognizer_text) else "error",
            detector_time_ms=detector_time_ms,
            recognizer_time_ms=recognizer_time_ms,
            vad_used=use_vad,
            vocab_boosted=vocab_boosted or None,
            wers_estimate=wers_estimate,
            quality_tier=quality_tier,
            coach_vocab_detected=coach_vocab_detected,
        )

    # ── Coach Vocabulary Check ────────────────────────────────────────────

    def _check_coach_vocabulary(self, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        return any(re.search(rf"\b{re.escape(term)}\b", low) for term in self.coach_vocabulary)

    # ── Language Model Reranking ──────────────────────────────────────────

    def _rerank_with_language_model(self, text: str):
        if not text:
            return "", 0.0
        normalized = self._normalize_text(text)
        coach_boost = 0.3 if self._check_coach_vocabulary(normalized) else 0.0
        word_count = len(normalized.split())
        length_score = min(1.0, word_count / 10)
        coherence_score = self._compute_coherence(normalized)
        final_confidence = round(
            max(0.0, min(100.0, coherence_score * 50 + length_score * 30 + coach_boost * 20)), 1
        )
        return normalized, final_confidence

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return text
        normalized = text.strip()
        if normalized and normalized[0].isalpha():
            normalized = normalized[0].upper() + normalized[1:]
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[.!?]+$", "", normalized).strip()
        return normalized

    @staticmethod
    def _compute_coherence(text: str) -> float:
        if not text:
            return 0.0
        words = re.findall(r"[a-zA-Z']+", text.lower())
        if not words:
            return 0.0
        valid = [w for w in words if len(w) >= 2 and w.isalpha()]
        coherence = len(valid) / len(words)
        if text[0].isupper():
            coherence += 0.1
        if text.rstrip().endswith((".", "!", "?")):
            coherence += 0.1
        return min(1.0, coherence)

    # ── WER Estimation ────────────────────────────────────────────────────

    def _estimate_wer(self, confidence: Optional[float], detector_confidence: float) -> Optional[float]:
        if confidence is None:
            return None
        if confidence >= 80:
            return round(max(0.0, 5.0 + (80 - confidence) * 0.1), 1)
        if confidence >= 50:
            return round(max(0.0, 8.0 + (50 - confidence) * 0.2), 1)
        if confidence >= 30:
            return round(max(0.0, 15.0 + (30 - confidence) * 0.3), 1)
        return round(max(0.0, 25.0 + (30 - confidence) * 0.5), 1)

    # ── Quality Classification ────────────────────────────────────────────

    @staticmethod
    def _classify_quality(wers_estimate: Optional[float], confidence: Optional[float]) -> str:
        if wers_estimate is None or confidence is None:
            return "fair"
        if wers_estimate <= 5.0 and confidence >= 80:
            return "excellent"
        if wers_estimate <= 8.0 and confidence >= 70:
            return "good"
        if wers_estimate <= 12.0 and confidence >= 50:
            return "fair"
        return "poor"


# ── Module-Level Convenience Function ────────────────────────────────────────

async def transcribe_pipeline(audio_bytes: bytes,
                              use_vad: bool = True,
                              detector_size: str = "tiny",
                              recognizer_size: str = "large-v3") -> ASRResult:
    pipeline = ASRPipeline(detector_size=detector_size, recognizer_size=recognizer_size)
    return await pipeline.transcribe(audio_bytes, use_vad=use_vad)
