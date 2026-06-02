"""
Correction service — parallel error-checking stream (fire-and-forget).
This module provides helpers used by ws.py's Stream C logic.
The 500ms timeout and Supabase persistence are handled in ws.py directly
to keep the hot path in one place.
"""

from services.llm import stream_correction_check


async def check_utterance(utterance: str) -> dict | None:
    """
    Run grammar correction check on a user utterance.
    Returns correction dict or None if no error or timeout.
    """
    return await stream_correction_check(utterance)
