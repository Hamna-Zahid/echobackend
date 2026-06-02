from typing import Literal, Optional
from pydantic import BaseModel


class CorrectionCard(BaseModel):
    error: bool
    error_type: Optional[Literal["TENSE", "ARTICLE", "PREPOSITION", "FILLER", "COHERENCE", "VOCAB"]] = None
    original: Optional[str] = None
    corrected: Optional[str] = None
    explanation: Optional[str] = None
    better_version: Optional[str] = None


class ErrorRecord(BaseModel):
    user_id: str
    session_id: str
    error_type: str
    original_utterance: str
    corrected_form: str
    explanation: Optional[str] = None
    recurrence_count: int = 1
    resolved: bool = False
