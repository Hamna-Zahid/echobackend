"""
ASR Quality Monitor — Real-time monitoring and metrics for the speech pipeline.

Tracks:
* Word Error Rate (WER) history
* Real-Time Factor (RTF) / latency
* Audio quality scores
* VAD statistics (true/false positives/negatives)
* Session quality classification
* Coach-specific term detection rates
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Set

import numpy as np

logger = logging.getLogger("echo.asr_quality_monitor")

# Quality thresholds (WER % boundaries)
QUALITY_TIERS = {
    "excellent": {"max_wer": 5.0, "min_confidence": 80, "description": "Near-perfect transcription"},
    "good": {"max_wer": 8.0, "min_confidence": 70, "description": "High quality, minor issues"},
    "fair": {"max_wer": 12.0, "min_confidence": 50, "description": "Acceptable with some errors"},
    "poor": {"max_wer": 20.0, "min_confidence": 0, "description": "Unreliable, significant errors"},
}

# Default monitoring window sizes
DEFAULT_WER_WINDOW = 100
DEFAULT_RTF_WINDOW = 100
DEFAULT_VAD_WINDOW = 1000


@dataclass
class WerMetric:
    """Word Error Rate measurement."""
    hypothesis: str          # ASR output
    reference: str           # Human-verified transcript
    wer: float               # Calculated WER (0-100+)
    session_id: str = "unknown"
    audio_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RtfMetric:
    """Real-Time Factor measurement."""
    session_id: str = "unknown"
    audio_duration: float = 0.0     # seconds of audio processed
    processing_time: float = 0.0    # seconds of CPU time
    rttf: float = 0.0               # Real-Time Factor = processing_time / audio_duration
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VadStats:
    """Voice Activity Detection statistics."""
    total_frames: int = 0
    true_positives: int = 0     # VAD said speech and it was speech
    false_positives: int = 0    # VAD said speech but it was silence
    false_negatives: int = 0    # VAD said silence but it was speech
    speech_frames: int = 0      # frames classified as speech
    silence_frames: int = 0     # frames classified as silence
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoachVocabStats:
    """Coach vocabulary detection statistics."""
    total_utterances: int = 0
    utterances_with_vocab: int = 0
    detected_terms: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def vocabulary_coverage(self) -> float:
        """Detected coach terms as a percentage of the full vocabulary."""
        from services.stt import COACH_VOCABULARY_TERMS
        total = len(COACH_VOCABULARY_TERMS) or 1
        return round(len(self.detected_terms) / total * 100, 1)


class ASRQualityMonitor:
    """
    Real-time quality monitoring for the ASR pipeline.

    Tracks cumulative metrics across sessions and provides quality
    classification, trends, and alerts.
    """

    def __init__(self,
                 wer_window: int = DEFAULT_WER_WINDOW,
                 rtf_window: int = DEFAULT_RTF_WINDOW,
                 vad_window: int = DEFAULT_VAD_WINDOW):
        # WER history (moving window)
        self.wer_history: Deque[WerMetric] = deque(maxlen=wer_window)

        # RTF history (moving window)
        self.rtf_history: Deque[RtfMetric] = deque(maxlen=rtf_window)

        # VAD statistics
        self.vad_stats = VadStats()

        # Coach vocabulary statistics
        self.coach_vocab = CoachVocabStats()

        # Session-level tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self._session_first_audio_ts: Optional[float] = None

        # Averages
        self._avg_wer: Optional[float] = None
        self._avg_rtf: Optional[float] = None
        self._quality_distribution: Dict[str, int] = {
            "excellent": 0, "good": 0, "fair": 0, "poor": 0,
        }

        logger.info(
            "ASRQualityMonitor initialized: WER window=%s, RTF window=%s, VAD window=%s",
            wer_window, rtf_window, vad_window,
        )

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_session(self, session_id: str) -> None:
        """Begin tracking a new session."""
        self.current_session_id = session_id
        self._t0 = time.time()
        self._session_start = time.time()
        logger.debug("quality session started: %s", session_id)

    def end_session(self) -> Dict[str, Any]:
        """End session tracking and return a summary of the session metrics."""
        if self.current_session_id is None:
            return {}

        session_id = self.current_session_id
        summary = {
            "session_id": session_id,
            "avg_wer": round(self._avg_wer, 1) if self._avg_wer is not None else None,
            "avg_rtf": round(self._avg_rtf, 3) if self._avg_rtf is not None else None,
            "quality_tier": self.get_quality_tier(),
            "vad_accuracy": self._compute_vad_accuracy(),
            "vocab_coverage": self.coach_vocab.vocabulary_coverage,
            "quality_distribution": dict(self._quality_distribution),
        }

        self.current_session_id = None
        self._rttf = None
        self._t_start = None
        return summary

    # Backwards-compatible alias used by the ASR pipeline.
    def update_metrics(self, hypothesis: str = "", reference: str = "",
                       latency: float = 0.0, audio_quality: float = 0.0,
                       estimated_wer: Optional[float] = None) -> None:
        if latency and audio_quality:
            pass
        if estimated_wer is not None:
            self.wer_history.append(WerMetric(
                hypothesis=hypothesis, reference=reference,
                wer=float(estimated_wer),
                session_id=self.current_session_id or "unknown",
                audio_duration=latency,
            ))
            self._avg_wer = self._mean_wer()

    # ── Recording ────────────────────────────────────────────────────────

    def record_utterance(self,
                         hypothesis: str,
                         reference: str = "",
                         audio_duration: float = 0.0,
                         detected_vocab: Optional[Set[str]] = None,
                         vad_result: Optional[VadStats] = None,
                         estimated_wer: Optional[float] = None) -> None:
        """Record metrics for a single user utterance."""
        # --- WER: real reference if provided, otherwise the estimated proxy ---
        wer: Optional[float] = None
        if reference:
            wer = self._compute_wer(hypothesis, reference)
        elif estimated_wer is not None:
            wer = float(estimated_wer)

        if wer is not None:
            self.wer_history.append(WerMetric(
                hypothesis=hypothesis,
                reference=reference,
                wer=wer,
                session_id=self.current_session_id or "unknown",
                audio_duration=audio_duration,
            ))
            self._avg_wer = self._mean_wer()
            tier = self._classify_quality_by_wer(wer)
            self._quality_distribution[tier] = self._quality_distribution.get(tier, 0) + 1

        # --- Coach vocabulary ---
        self.coach_vocab.total_utterances += 1
        if detected_vocab:
            self.coach_vocab.utterances_with_vocab += 1
            self.coach_vocab.detected_terms.update(detected_vocab)

        # --- RTF: time since the last utterance was recorded ---
        if audio_duration and audio_duration > 0:
            rttf = audio_duration  # placeholder if no wall-clock reference
            self._rttf = rttf
            self.rtf_history.append(RtfMetric(
                session_id=self.current_session_id or "unknown",
                audio_duration=audio_duration,
                processing_time=rttf,
                rttf=rttf,
            ))
            self._avg_rtf = self._mean_rtf()

        # --- VAD ---
        if vad_result is not None:
            self.vad_stats.total_frames += vad_result.total_frames
            self.vad_stats.true_positives += vad_result.true_positives
            self.vad_stats.false_positives += vad_result.false_positives
            self.vad_stats.false_negatives += vad_result.false_negatives
            self.vad_stats.speech_frames += vad_result.speech_frames
            self.vad_stats.silence_frames += vad_result.silence_frames

    def record_rtf(self, audio_duration: float, processing_time: float) -> None:
        """Record a real-time-factor measurement (wall clock vs audio length)."""
        if audio_duration <= 0:
            return
        rttf = processing_time / audio_duration
        self.rtf_history.append(RtfMetric(
            session_id=self.current_session_id or "unknown",
            audio_duration=audio_duration,
            processing_time=processing_time,
            rttf=rttf,
        ))
        self._avg_rtf = self._mean_rtf()

    def record_vocab_detection(self, terms: Set[str]) -> None:
        """Record coach vocabulary terms detected in the latest utterance."""
        if terms:
            self.coach_vocab.detected_terms.update(terms)

    def update_vad_frame(self, is_speech: bool, actual_is_speech: bool) -> None:
        """Update VAD statistics for a single frame."""
        self.vad_stats.total_frames += 1
        if is_speech and actual_is_speech:
            self.vad_stats.true_positives += 1
            self.vad_stats.speech_frames += 1
        elif is_speech and not actual_is_speech:
            self.vad_stats.false_positives += 1
            self.vad_stats.speech_frames += 1
        elif not is_speech and actual_is_speech:
            self.vad_stats.false_negatives += 1
        else:
            self.vad_stats.silence_frames += 1

    # ── Helpers ──────────────────────────────────────────────────────────

    def _mean_wer(self) -> Optional[float]:
        if not self.wer_history:
            return None
        return sum(w.wer for w in self.wer_history) / len(self.wer_history)

    def _mean_rtf(self) -> Optional[float]:
        if not self.rtf_history:
            return None
        return sum(r.rttf for r in self.rtf_history) / len(self.rtf_history)

    def _compute_wer(self, hypothesis: str, reference: str) -> float:
        """Levenshtein-based WER = (S + D + I) / N * 100."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 0.0 if not hyp_words else 100.0
        m, n = len(hyp_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp_words[i - 1] == ref_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return round(min(100.0, dp[m][n] / len(ref_words) * 100), 1)

    def _classify_quality_by_wer(self, wer: Optional[float]) -> str:
        if wer is None:
            return "fair"
        for tier, thresholds in QUALITY_TIERS.items():
            if wer <= thresholds["max_wer"]:
                return tier
        return "poor"

    def get_quality_tier(self) -> str:
        return self._classify_quality_by_wer(self._avg_wer)

    def _compute_vad_accuracy(self) -> float:
        total = self.vad_stats.total_frames
        if total == 0:
            return 0.0
        # Accuracy = (TP + TN) / total. We don't track TN directly; derive it.
        tn = max(0, total - self.vad_stats.speech_frames - self.vad_stats.false_negatives)
        correct = self.vad_stats.true_positives + tn
        return round(min(100.0, correct / total * 100), 1)

    # ── Reporting ────────────────────────────────────────────────────────

    def get_trends(self) -> Dict[str, Any]:
        wer_list = [w.wer for w in self.wer_history]
        n = len(wer_list)
        trend = "stable"
        if n > 1:
            x = list(range(n))
            mx = sum(x) / n
            my = sum(wer_list) / n
            denom = (n * sum(v ** 2 for v in x) - sum(x) ** 2) or 1
            slope = (n * sum(x[i] * wer_list[i] for i in range(n)) - sum(x) * sum(wer_list)) / denom
            if abs(slope) < 0.5:
                trend = "stable"
            elif slope < 0:
                trend = "improving"
            else:
                trend = "degrading"
        return {
            "avg_wer": round(sum(wer_list) / n, 1) if n else 0.0,
            "min_wer": min(wer_list) if wer_list else 0.0,
            "max_wer": max(wer_list) if wer_list else 0.0,
            "trend": trend,
            "rtf_avg": round(self._avg_rtf, 3) if self._avg_rtf is not None else 0.0,
            "quality_distribution": dict(self._quality_distribution),
            "total_utterances": n,
        }

    def get_report(self) -> Dict[str, Any]:
        return {
            "summary": {
                "avg_wer": round(self._avg_wer, 1) if self._avg_wer is not None else 0.0,
                "avg_rtf": round(self._avg_rtf, 3) if self._avg_rtf is not None else 0.0,
                "quality_tier": self.get_quality_tier(),
                "total_utterances": len(self.wer_history),
                "quality_distribution": dict(self._quality_distribution),
            },
            "wer_trends": self.get_trends(),
            "vad_stats": {
                "total_frames": self.vad_stats.total_frames,
                "accuracy": self._compute_vad_accuracy(),
                "true_positives": self.vad_stats.true_positives,
                "false_positives": self.vad_stats.false_positives,
                "false_negatives": self.vad_stats.false_negatives,
            },
            "coach_vocabulary": {
                "total_utterances": self.coach_vocab.total_utterances,
                "utterances_with_vocab": self.coach_vocab.utterances_with_vocab,
                "vocabulary_coverage": self.coach_vocab.vocabulary_coverage,
                "detected_terms": sorted(self.coach_vocab.detected_terms),
            },
            "timestamp": datetime.now().isoformat(),
        }
