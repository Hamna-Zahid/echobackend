"""
Feedback engine — turns a finished practice session into a structured scorecard.

This is the part of the product people actually pay for: accurate, specific
coaching. Delivery numbers (WPM, filler counts) are measured from the real audio
in stt.py and passed in here, so they are facts — the LLM only does the
qualitative analysis (STAR, structure, model-answer rewrites).
"""

from prompts.feedback import FEEDBACK_SYSTEM
from services.llm import generate_feedback_json
from services.gemma_audio import format_delivery_notes


def compute_delivery_metrics(
    total_user_words: int,
    total_user_seconds: float,
    filler_hits: dict,
) -> dict:
    """Aggregate the per-turn numbers collected during the session into deliverable stats."""
    minutes = max(total_user_seconds / 60.0, 1e-6)
    wpm = int(round(total_user_words / minutes)) if total_user_words else 0
    filler_count = sum(filler_hits.values())
    top = sorted(filler_hits.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "pace_wpm": wpm,
        "filler_count": filler_count,
        "top_fillers": [w for w, _ in top[:3]],
        "spoken_seconds": int(total_user_seconds),
        "word_count": total_user_words,
    }


def _format_transcript(turns: list[dict], mode: str) -> str:
    speaker = {"interview": ("Interviewer", "Candidate"),
               "speech": ("Coach", "Speaker"),
               "professional": ("Colleague", "You")}.get(mode, ("Coach", "You"))
    lines = []
    for t in turns:
        who = speaker[0] if t.get("role") == "assistant" else speaker[1]
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    return "\n".join(lines)


async def generate_session_feedback(
    mode: str,
    turns: list[dict],
    metrics: dict,
    delivery_notes: list[dict] | None = None,
    usage_sink: dict | None = None,
) -> dict | None:
    """
    Build the structured coaching report. Returns the report dict, or None if
    there wasn't enough spoken content to assess.

    `delivery_notes` are optional per-answer audio judgments from Gemma 4
    (confidence/tone/hesitancy heard in the actual recordings) — when present
    they ground the clarity/confidence scores in real audio, not guesses.
    """
    transcript = _format_transcript(turns, mode)
    # Need at least some candidate speech to score.
    if metrics.get("word_count", 0) < 15:
        return None

    audio_notes = format_delivery_notes(delivery_notes or [])
    audio_block = f"{audio_notes}\n\n" if audio_notes else ""

    payload = (
        f"SESSION MODE: {mode}\n\n"
        f"DELIVERY METRICS (measured from audio — use these exact numbers):\n"
        f"  words_per_minute = {metrics['pace_wpm']}\n"
        f"  filler_word_count = {metrics['filler_count']}\n"
        f"  top_fillers = {metrics['top_fillers']}\n"
        f"  total_spoken_seconds = {metrics['spoken_seconds']}\n\n"
        f"{audio_block}"
        f"━━━ TRANSCRIPT ━━━\n{transcript}\n━━━ END TRANSCRIPT ━━━\n\n"
        f"Produce the JSON scorecard now."
    )

    report = await generate_feedback_json(FEEDBACK_SYSTEM, payload, usage_sink=usage_sink)
    if not report:
        return None

    # Trust our measured numbers over anything the model wrote.
    delivery = report.setdefault("delivery", {})
    delivery["pace_wpm"] = metrics["pace_wpm"]
    delivery["filler_count"] = metrics["filler_count"]
    if metrics["top_fillers"]:
        delivery["top_fillers"] = metrics["top_fillers"]
    return report
