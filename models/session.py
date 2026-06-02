from typing import Optional
from pydantic import BaseModel, Field


class SessionState(BaseModel):
    user_id: str
    mode: str  # diagnostic | training | interview | conversation | professional | accent
    is_diagnostic_complete: bool = False
    diagnostic_turn_count: int = 0
    phoneme_unit: int = 0
    current_topic: str = "career"
    silence_timer_start: Optional[float] = None
    error_count_this_session: int = 0
    cefr_level: Optional[str] = None
    top_recurring_errors: list = []
    tts_is_speaking: bool = False
    session_db_id: str = ""
    preferred_accent: str = "american"
    conversation_history: list = []
    session_transcript: str = ""
    session_started_at: float = 0.0
    user_memory: dict = Field(default_factory=dict)
    pronunciation_scores: list = []
