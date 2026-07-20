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
    memory_context: str = "",
    learner_name: str = "",
) -> str:
    context = build_coaching_context(cefr, focus, top_errors)
    memory_block = f"\n\n{memory_context}" if memory_context else ""
    name_line = f"\nTHE LEARNER'S NAME: {learner_name} — use it naturally." if learner_name else ""

    # A returning learner must never get the cold-open script; the memory block
    # tells the model to resume the relationship instead.
    if memory_context:
        opening = (
            "Begin the call now: greet them by name like a coach who already knows them, "
            "refer to something specific you remember from last time, and ask what they'd "
            "like to talk about or work on today. Do NOT introduce yourself again and do "
            "NOT use a generic scripted opener."
        )
    else:
        opening = (
            "Begin the call now. This is your FIRST EVER conversation with this learner — "
            "you have never spoken before, so do NOT say things like 'again', 'as always' "
            "or 'good to catch up'. Introduce yourself as Echo, then ask ONE easy opening "
            "question that invites them to say what they'd like to talk about or practise today."
        )

    return f"""{context}{memory_block}
{name_line}
ACCENT: model {accent} English pronunciation and vocabulary.

{opening}"""
