"""
VAD service — Voice Activity Detection with adaptive thresholds.

Provides real-time speech detection for the coach use case:
- Adaptive energy thresholds based on environment
- Temporal smoothing with majority vote
- Environment-based configuration (quiet/office/mobile/noisy)
- Free-tier compatible — no GPU, no paid APIs

Design constraints:
* PCM F32LE, 16kHz mono input (standard for Whisper)
* Works with audio chunks from WebSocket audio_buffer
* Must never block the event loop — fire-and-forget friendly
* Thresholds tuned for coaching / conversational speech
"""

import numpy as np
from typing import Optional
from collections import deque


# ── Environment-based VAD thresholds ─────────────────────────────────────────
# These are dBFS (decibels full scale) references. Lower = more sensitive.
# Measured relative to 1.0 = full-scale sine wave V RMS.

VAD_CONFIG = {
    "quiet_room": {
        "energy_threshold_dbfs": -35,    # Very quiet, stable environment
        "silence_ms": 700,                # How long before we call it silence
        "max_speech_gap_ms": 1500,       # Max gap between speech fragments to still group them
        "pre_silence_ms": 100,           # Pre-roll buffer before detection
    },
    "office": {
        "energy_threshold_dbfs": -30,
        "silence_ms": 600,
        "max_speech_gap_ms": 1200,
        "pre_silence_ms": 80,
    },
    "mobile": {
        "energy_threshold_dbfs": -25,
        "silence_ms": 500,
        "max_speech_gap_ms": 1000,
        "pre_silence_ms": 60,
    },
    "noisy": {
        "energy_threshold_dbfs": -20,
        "silence_ms": 300,
        "max_speech_gap_ms": 800,
        "pre_silence_ms": 40,
    },
}

DEFAULT_CONFIG = "mobile"  # Most coaching happens on mobile devices


# ── VAD Result ───────────────────────────────────────────────────────────────

class VadResult:
    """Result from a single VAD evaluation."""
    def __init__(self, is_speech: bool, rms: float, dbfs: float,
                 stable: bool, duration_ms: int):
        self.is_speech = is_speech
        self.rms = rms                     # Root-mean-square amplitude
        self.dbfs = dbfs                     # dBFS value
        self.stable = stable                 # Whether above threshold for minimum duration
        self.duration_ms = duration_ms       # How long it's been above threshold


# ── ImprovedVAD Class ────────────────────────────────────────────────────────

class ImprovedVAD:
    """
    Voice Activity Detection with adaptive thresholds and temporal smoothing.

    Features:
    - Adaptive energy threshold based on running background noise estimate
    - Majority-vote smoothing over recent frames (reduces false positives)
    - Environment presets for different recording conditions
    - Silence duration tracking for turn-end detection
    - Pre-roll buffer so we don't clip the start of speech

    Input: numpy array of PCM F32LE samples, 16kHz mono expected
    Output: VadResult with detection decision
    """

    def __init__(self, config_name: str = DEFAULT_CONFIG,
                 sample_rate: int = 16000,
                 frame_ms: int = 20):
        """
        Args:
            config_name: "quiet_room" | "office" | "mobile" | "noisy"
            sample_rate: Audio sample rate in Hz (Whisper expects 16kHz)
            frame_ms: Analysis frame duration in ms (20ms = 80 frames/sec)
        """
        self.config = VAD_CONFIG[config_name]
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

        # Adaptive threshold tracking
        self.background_rms: Optional[float] = None  # Running estimate of noise floor
        self.noise_time_ms: float = 0.0               # How long we've been in silence
        self.speech_time_ms: float = 0.0              # How long we've been in speech

        # Temporal smoothing buffer (majority vote)
        self.recent_decisions: deque = deque(maxlen=5)  # Last 5 frames = ~100ms

        # State tracking
        self.is_speech: bool = False
        self.speech_start_ms: Optional[float] = None  # When current speech block started
        self.last_silence_ms: Optional[float] = None  # When last silence block started

    # ── Public API ─────────────────────────────────────────────────────────

    def _to_float32(self, pcm_chunk) -> np.ndarray:
        return _to_float32(pcm_chunk)

    @staticmethod
    def _to_float32_static(pcm_chunk) -> np.ndarray:
        return _to_float32(pcm_chunk)

    def update_noise_profile(self, pcm_chunk) -> None:
        """
        Update the background noise estimate from a silence chunk.
        Call this when you know the current audio is noise (e.g. at session start).
        """
        pcm_chunk = self._to_float32(pcm_chunk)
        if len(pcm_chunk) < self.frame_samples:
            return

        # Compute RMS of this chunk
        rms = float(np.sqrt(np.mean(pcm_chunk**2)))

        # Exponential moving average of background noise
        if self.background_rms is None:
            self.background_rms = rms
        else:
            # EMA: 90% existing, 10% new — smooths over time
            self.background_rms = 0.9 * self.background_rms + 0.1 * rms

    def reset(self) -> None:
        """Reset VAD state (call at session start or mode change)."""
        self.background_rms = None
        self.noise_time_ms = 0.0
        self.speech_time_ms = 0.0
        self.recent_decisions.clear()
        self.is_speech = False
        self.speech_start_ms = None
        self.last_silence_ms = None

    def detect_speech(self, pcm_chunk) -> Optional[VadResult]:
        """
        Detect if the given PCM chunk contains speech.

        Args:
            pcm_chunk: PCM audio — raw int16 bytes, int16 array, or float32 array
                Expected: 16kHz mono, but will work with whatever size is given

        Returns:
            VadResult if enough data for a decision, None if chunk too small
            None means "don't make a decision yet, wait for more data"
        """
        pcm_chunk = self._to_float32(pcm_chunk)
        if len(pcm_chunk) < self.frame_samples:
            return None

        # Compute RMS and dBFS
        rms = float(np.sqrt(np.mean(pcm_chunk**2)))
        # Avoid log(0)
        eps = 1e-10
        dbfs = 20.0 * np.log10(rms + eps)

        # Adaptive threshold: start from config, adjust based on background
        if self.background_rms is not None and self.background_rms > 0:
            # Threshold is X dB above the estimated noise floor
            threshold_dbfs = self.background_rms + self.config["energy_threshold_dbfs"]
        else:
            # Use config threshold assuming reasonable audio level
            threshold_dbfs = self.config["energy_threshold_dbfs"]

        # Determine raw decision: is current frame above threshold?
        raw_is_speech = dbfs > threshold_dbfs

        # Add to smoothing buffer
        self.recent_decisions.append(raw_is_speech)

        # Majority vote: need at least 3 of recent frames to agree
        # With maxlen=5, we need 3+ to be consistent
        if len(self.recent_decisions) >= 3:
            ones = sum(self.recent_decisions)
            stable = ones >= 3  # 3 out of 5 agree
        else:
            stable = raw_is_speech  # Not enough data yet, use raw

        # Track speech/silence duration
        current_ms = self.frame_ms
        if stable and not self.is_speech:
            # Speech just started
            self.is_speech = True
            self.speech_start_ms = current_ms  # Simplified tracking
        elif not stable and self.is_speech:
            # Speech just stopped
            duration = (current_ms if self.speech_start_ms is None
                        else current_ms - self.speech_start_ms)
            self.is_speech = False
            self.speech_start_ms = None
            self.last_silence_ms = current_ms

        # Return result
        return VadResult(
            is_speech=stable,
            rms=rms,
            dbfs=dbfs,
            stable=stable,
            duration_ms=int(self.frame_ms) if stable else 0
        )

    # ── Convenience: check if speech has paused long enough ──────────────────

    def speech_paused_long_enough(self, current_ms: int, pause_ms_threshold: Optional[int] = None) -> bool:
        """
        Check if we've been in silence long enough to consider the speech paused.

        Args:
            current_ms: Current timestamp in ms (e.g. time.time() * 1000)
            pause_ms_threshold: Override the default silence_ms from config

        Returns:
            True if speech has been paused long enough to end the turn
        """
        if not self.is_speech:
            # Already silent — check if we've been quiet long enough
            threshold = pause_ms_threshold or self.config["silence_ms"]
            if self.last_silence_ms is not None:
                elapsed = current_ms - self.last_silence_ms
                return elapsed >= threshold
            return True  # Were already silent

        # Still in speech — not paused yet
        return False

    # ── Convenience: get current environment config ──────────────────────────

    @staticmethod
    def get_config_summary(config_name: str = DEFAULT_CONFIG) -> dict:
        """Human-readable summary of a VAD config."""
        c = VAD_CONFIG[config_name]
        return {
            "environment": config_name,
            "energy_threshold_dbfs": c["energy_threshold_dbfs"],
            "silence_ms": c["silence_ms"],
            "max_speech_gap_ms": c["max_speech_gap_ms"],
            " sensitivity": "High" if c["energy_threshold_dbfs"] < -30 else "Medium" if c["energy_threshold_dbfs"] < -25 else "Aggressive",
        }


# ── Module-level convenience function ────────────────────────────────────────

def detect_speech_simple(pcm_chunk, config_name: str = DEFAULT_CONFIG,
                         sample_rate: int = 16000) -> Optional[VadResult]:
    """
    Quick VAD check without maintaining state.
    """
    vad = ImprovedVAD(config_name=config_name, sample_rate=sample_rate)
    return vad.detect_speech(pcm_chunk)