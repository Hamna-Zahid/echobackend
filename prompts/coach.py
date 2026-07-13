"""
Echo's single adaptive coaching persona.

The system prompt is assembled from the Coaching Knowledge Base (coaching_kb.py)
so the model is always grounded in: who Echo is, what good coaching looks like at
THIS learner's CEFR level, the session focus, and the learner's recurring weak
spots. This is the app's "retrieval-augmented" coaching context — the right
methodology is selected per-learner and injected, rather than relying on a
generic chat model.
"""

from prompts.coaching_kb import build_coaching_context


def build_coach_prompt(
    cefr: str = "B1",
    accent: str = "american",
    focus: str = "general",
    top_errors: list[str] | None = None,
) -> str:
    context = build_coaching_context(cefr, focus, top_errors)
    return f"""{context}

ACCENT: model {accent} English pronunciation and vocabulary.

Begin the call now: greet the learner warmly by opening the conversation (introduce yourself as Echo if it feels natural), and ask ONE easy, personal opening question to get them talking."""
