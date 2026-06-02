"""
Post-session phoneme analysis — Wav2Vec2-Phoneme (facebook/wav2vec2-lv-60-espeak-cv-ft).

Runs EXCLUSIVELY after a session ends on the post-session slow track.
Never loaded during live conversation — avoids competing with runtime memory.

On 8 GB RAM:
  Model size: ~1.2 GB on disk, ~900 MB in RAM
  Loaded on demand, unloaded after batch processing
  Processing: ~0.5s per 10s of audio on CPU

Pipeline:
  stored audio buffers → Wav2Vec2 → IPA phoneme string → diff vs expected → mismatch list

The mismatch list feeds into the LLM report generator (llm.generate_session_report)
to produce specific, actionable pronunciation coaching tips.
"""

import io
import os
from typing import Optional

import numpy as np

# ── IPA phoneme reference for common errors ───────────────────────────────────

# Maps IPA phoneme → common native-language substitutions and coaching tips
PHONEME_COACHING = {
    "θ": {
        "common_errors": ["t", "s", "f", "d"],
        "tip": "Place your tongue lightly between your front teeth and breathe out. Like 'th' in 'think'.",
        "example_words": ["think", "three", "nothing", "health"],
    },
    "ð": {
        "common_errors": ["d", "z", "v"],
        "tip": "Same tongue position as /θ/ but vibrate your vocal cords. Like 'th' in 'this'.",
        "example_words": ["this", "the", "there", "mother"],
    },
    "ŋ": {
        "common_errors": ["n", "ŋɡ"],
        "tip": "Raise the back of your tongue to the soft palate. No 'g' sound at the end — just 'ng'.",
        "example_words": ["thinking", "working", "going", "singing"],
    },
    "æ": {
        "common_errors": ["e", "ɛ", "a"],
        "tip": "Open your mouth wide, like you're saying 'a' but push your tongue forward and down.",
        "example_words": ["cat", "bad", "have", "that"],
    },
    "ɪ": {
        "common_errors": ["iː", "e"],
        "tip": "Short and relaxed — not the long 'ee' sound. Like the 'i' in 'bit', not 'beat'.",
        "example_words": ["bit", "sit", "fill", "still"],
    },
    "ʊ": {
        "common_errors": ["uː", "o"],
        "tip": "Short and relaxed 'oo'. Like 'u' in 'put', not 'pool'.",
        "example_words": ["put", "book", "could", "should"],
    },
    "ʒ": {
        "common_errors": ["z", "ʃ", "dʒ"],
        "tip": "Like 'z' but with your lips slightly rounded, like the 's' in 'measure'.",
        "example_words": ["measure", "vision", "pleasure", "usual"],
    },
    "r": {
        "common_errors": ["ɾ", "l", "w"],
        "tip": "Curl your tongue tip back slightly without touching the roof. Don't roll it.",
        "example_words": ["red", "right", "try", "very"],
    },
    "v": {
        "common_errors": ["b", "w", "f"],
        "tip": "Upper teeth touch lower lip, then vibrate. Like 'f' but voiced.",
        "example_words": ["very", "have", "voice", "five"],
    },
    "w": {
        "common_errors": ["v", "b"],
        "tip": "Round your lips tightly first, then open into the vowel. No teeth.",
        "example_words": ["word", "want", "will", "twelve"],
    },
}

# ── Model loading (lazy, singleton) ──────────────────────────────────────────

_processor = None
_model = None
_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def _load_model():
    """Load Wav2Vec2 model (once per process). ~900 MB RAM."""
    global _processor, _model
    if _processor is None or _model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch

        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _processor, _model


def _unload_model():
    """Free model from RAM after batch processing."""
    global _processor, _model
    _processor = None
    _model = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── Audio utilities ───────────────────────────────────────────────────────────

def _pcm_f32le_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM F32LE bytes to a normalised float32 numpy array."""
    return np.frombuffer(audio_bytes, dtype=np.float32).copy()


def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16kHz required by Wav2Vec2."""
    if orig_sr == 16000:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
    except ImportError:
        # Manual linear interpolation fallback
        ratio = 16000 / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_pronunciation_batch(
    audio_segments: list[dict],
    orig_sample_rate: int = 24000,
) -> list[dict]:
    """
    Batch phoneme analysis for a list of recorded audio segments.

    Each segment: {"audio_bytes": bytes, "expected_text": str, "timestamp": float}
    Returns: list of {"target_word": str, "detected_ipa": str, "target_ipa": str,
                       "coaching_tip": str, "severity": "minor"|"major"}

    Called post-session — can take several seconds, never called during live conversation.
    """
    if not audio_segments:
        return []

    try:
        import torch
        processor, model = _load_model()
        results = []

        for seg in audio_segments:
            audio_bytes = seg.get("audio_bytes")
            expected_text = seg.get("expected_text", "")
            if not audio_bytes or not expected_text:
                continue

            try:
                audio = _pcm_f32le_to_numpy(audio_bytes)
                audio_16k = _resample_to_16k(audio, orig_sample_rate)

                inputs = processor(
                    audio_16k,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                with torch.no_grad():
                    logits = model(inputs.input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                detected_ipa = processor.batch_decode(predicted_ids)[0]

                # Compare against expected phonemes
                mismatches = _find_mismatches(detected_ipa, expected_text)
                results.extend(mismatches)

            except Exception:
                continue

        return results

    except ImportError:
        return []  # transformers/torch not installed — skip silently
    finally:
        _unload_model()


def _find_mismatches(detected_ipa: str, expected_text: str) -> list[dict]:
    """
    Compare detected IPA against known phoneme coaching map.
    Returns a list of mismatch records with coaching tips.
    """
    results = []
    detected_phones = detected_ipa.split()

    for target_phoneme, coaching in PHONEME_COACHING.items():
        for error_phoneme in coaching["common_errors"]:
            if error_phoneme in detected_phones and target_phoneme not in detected_phones:
                # Likely mismatch: expected target phoneme, heard error phoneme
                results.append({
                    "target_word": _find_target_word(expected_text, coaching["example_words"]),
                    "detected_ipa": error_phoneme,
                    "target_ipa": target_phoneme,
                    "coaching_tip": coaching["tip"],
                    "example_words": coaching["example_words"][:2],
                    "severity": "major" if target_phoneme in ["θ", "ð", "r", "v"] else "minor",
                })
                break  # one mismatch per phoneme pair

    return results


def _find_target_word(text: str, example_words: list[str]) -> str:
    """Find the most relevant example word that appears in the expected text."""
    text_lower = text.lower()
    for word in example_words:
        if word in text_lower:
            return word
    return example_words[0] if example_words else text.split()[0] if text.split() else ""


# ── Quick single-utterance analysis (for accent mode live feedback) ───────────

def quick_phoneme_check(audio_bytes: bytes, orig_sample_rate: int = 24000) -> Optional[str]:
    """
    Fast phoneme extraction for a single utterance.
    Returns raw IPA string. Used in accent mode for near-real-time coaching.
    Load/unload model each call — only use when the latency budget allows.
    """
    try:
        import torch
        processor, model = _load_model()
        audio = _pcm_f32le_to_numpy(audio_bytes)
        audio_16k = _resample_to_16k(audio, orig_sample_rate)
        inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    except Exception:
        return None
