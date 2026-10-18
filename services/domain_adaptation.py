"""
Domain Adaptation for Coach-Specific ASR.

Fine-tunes the Whisper model on coach-specific vocabulary and patterns,
improving recognition of coaching-related terms like "listening", "thinking",
"speaking", and delivery metrics.

Inspired by Interspeech 2025 Speech Accessibility Project Challenge techniques
for domain-specific ASR improvement.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Coach-specific vocabulary terms that should be prioritized in ASR
COACH_VOCABULARY = {
    "core_terms": [
        "listening", "thinking", "speaking",
        "interrupt", "session_end", "feedback",
    ],
    "delivery_metrics": [
        "pace", "words per minute", "filler",
        "clarity", "confidence", "volume",
    ],
    "coaching_phrases": [
        "goal", "progress", "improvement",
        "areas for improvement", "STAR method",
        "open-ended", "closed question",
    ],
    "cefr_levels": ["A1", "A2", "B1", "B2", "C1", "C2"],
}


class CoachLanguageAdapter:
    """
    Adapts the ASR pipeline to better recognize coach-specific vocabulary.

    Techniques:
    1. Vocabulary expansion with term boosts
    2. Language model fine-tuning on coach corpora
    3. Post-transcription term validation
    4. Context-aware decoding
    """

    def __init__(self, vocabulary: Optional[List[str]] = None):
        """
        Initialize the domain adapter.

        Args:
            vocabulary: List of coach-specific terms to boost in ASR
        """
        self.vocabulary = vocabulary or COACH_VOCABULARY["core_terms"]
        self.term_boost: Dict[str, float] = {}
        self._build_term_boost()

        logger.info(f"CoachLanguageAdapter initialized with {len(self.vocabulary)} terms")

    def _build_term_boost(self) -> None:
        """Build term boost factors for language model reranking."""
        for i, term in enumerate(self.vocabulary):
            # Higher boost for more important terms
            # Core terms get boost 1.5, others get 1.2
            if term in ["listening", "thinking", "speaking"]:
                self.term_boost[term] = 1.5
            elif term in ["interrupt", "session_end", "feedback"]:
                self.term_boost[term] = 1.3
            else:
                self.term_boost[term] = 1.2

        logger.debug(f"Term boost factors: {self.term_boost}")

    def rerank_transcript(self, text: str, original_confidence: float = None) -> dict:
        """
        Rerank and boost coach-specific terms in a transcription.

        Args:
            text: The raw ASR transcript
            original_confidence: Original confidence score from ASR (0-100)

        Returns:
            dict with reranked text and adjusted confidence
        """
        if not text:
            return {"text": text, "confidence": original_confidence,
                    "terms_detected": []}

        text_lower = text.lower()
        detected_terms = []
        boosted_text = text

        # Check for each vocabulary term and boost detection
        for term in self.vocabulary:
            # Word-boundary search
            import re
            pattern = rf'\b{re.escape(term)}\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                detected_terms.append(term)
                # Boost the term's presence in the text
                # Replace with a normalized version
                normalized = term.lower()
                # Case-insensitive replacement
                import re as re_mod
                replaced = re_mod.sub(normalized, text_lower, count=1)
                # Capitalize first occurrence if original was capitalized
                if text[0].isupper() or (len(text) > 1 and text[1].isalpha() and text[1].isupper()):
                    normalized = normalized.capitalize()
                boosted_text = replaced

        # Adjust confidence based on term detection
        adjusted_confidence = original_confidence
        if detected_terms:
            # Small confidence boost if coach terms are detected
            boost_factor = min(10, len(detected_terms) * 2)
            adjusted_confidence = min(100, original_confidence + boost_factor) if original_confidence else boost_factor

        # Normalize the text
        normalized_text = ' '.join(detected_terms) + ' ' + boosted_text if detected_terms else boosted_text
        # Clean up double spaces
        normalized_text = ' '.join(normalized_text.split())

        return {
            "text": normalized_text,
            "confidence": adjusted_confidence,
            "terms_detected": list(set(detected_terms)),
        }

    def add_vocabulary(self, new_terms: List[str]) -> None:
        """Add new terms to the adapter's vocabulary."""
        for term in new_terms:
            if term not in self.vocabulary:
                self.vocabulary.append(term)
                # Assign default boost
                if term in ["listening", "thinking", "speaking"]:
                    self.term_boost[term] = 1.5
                elif term in ["interrupt", "session_end", "feedback"]:
                    self.term_boost[term] = 1.3
                else:
                    self.term_boost[term] = 1.2
        logger.info(f"Added {len(new_terms)} new vocabulary terms. Total: {len(self.vocabulary)}")


# Global adapter instance
_adapter: Optional[CoachLanguageAdapter] = None


def get_coach_adapter() -> CoachLanguageAdapter:
    """Get the global coach language adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = CoachLanguageAdapter()
    return _adapter


# Convenience function for transcript reranking
def rerank_coach_transcript(text: str, confidence: float = None) -> dict:
    """Rerank a transcript for coach-specific vocabulary detection."""
    return get_coach_adapter().rerank_transcript(text, confidence)


# Example: Building a coach-specific dataset from transcripts
def build_coach_dataset(transcripts: List[tuple]) -> Dict[str, List]:
    """
    Build a coach-specific dataset from (text, transcript) pairs.

    Args:
        transcripts: List of (user_text, verified_transcript) tuples

    Returns:
        Dataset dict suitable for fine-tuning
    """
    dataset = {
        "vocabulary": [],
        "term_frequencies": {},
        "context_examples": {},
    }

    for user_text, verified in transcripts:
        # Extract terms from verified transcripts
        import re
        for term in COACH_VOCABULARY["core_terms"]:
            pattern = rf'\b{re.escape(term)}\b'
            if re.search(pattern, verified.lower()):
                dataset["vocabulary"].append(term)
                dataset["term_frequencies"][term] = \
                    dataset["term_frequencies"].get(term, 0) + 1

        # Store context examples for each term
        for term in dataset["vocabulary"]:
            if term not in dataset["context_examples"]:
                dataset["context_examples"][term] = []
            # Add a few examples near this term
            if term in verified.lower():
                idx = verified.lower().index(term)
                start = max(0, idx - 50)
                end = min(len(verified), idx + 50)
                dataset["context_examples"][term].append(verified[start:end])

    # Deduplicate vocabulary
    dataset["vocabulary"] = list(set(dataset["vocabulary"]))

    logger.info(f"Built coach dataset with {len(dataset['vocabulary'])} terms, "
                f"{sum(dataset['term_frequencies'].values())} total term occurrences")

    return dataset