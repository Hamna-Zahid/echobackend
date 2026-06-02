"""
Curriculum engine — structured coaching plans per session.

Each coaching session has a dynamic plan that tells the LLM:
  - What skill we are targeting this turn
  - Which phase of the session we are in
  - What the learner has already demonstrated
  - What to push for next

This context is injected at the top of every system prompt turn,
making the AI genuinely goal-directed instead of just "having a chat".

Session phases:
  WARMUP      (turns 1-2)  : Easy opener, gauge current state, build comfort
  CORE        (turns 3-5)  : Target the session's primary skill focus
  CHALLENGE   (turns 6-8)  : Push slightly above CEFR comfort zone
  APPLY       (turns 9+)   : Spontaneous use of session targets in new context
"""

from typing import Optional


# ── Skill maps per CEFR level ─────────────────────────────────────────────────

CEFR_SKILLS = {
    "A1": {
        "grammar":    ["simple present", "basic questions", "there is/are"],
        "vocabulary": ["numbers, colors, greetings", "family", "food and places"],
        "tasks":      ["name and describe things", "simple personal facts"],
    },
    "A2": {
        "grammar":    ["simple past", "present continuous", "modal verbs: can/must"],
        "vocabulary": ["daily routines", "work and study", "travel basics"],
        "tasks":      ["narrate past events", "express preferences with because"],
    },
    "B1": {
        "grammar":    ["present perfect vs simple past", "first conditional", "passive voice"],
        "vocabulary": ["discourse markers", "topic-specific collocations", "phrasal verbs"],
        "tasks":      ["describe trends", "give opinions with reasons", "narrate a story"],
    },
    "B2": {
        "grammar":    ["second and third conditional", "reporting verbs", "complex noun phrases"],
        "vocabulary": ["academic/formal register", "hedging language", "idioms in context"],
        "tasks":      ["argue a position", "compare and contrast", "speculate about causes"],
    },
    "C1": {
        "grammar":    ["inversion for emphasis", "cleft sentences", "complex relative clauses"],
        "vocabulary": ["nuanced synonyms", "connotation and register", "culture-specific references"],
        "tasks":      ["defend a nuanced position", "analyse cause and effect", "use irony and subtext"],
    },
    "C2": {
        "grammar":    ["full native range — focus on precision and style"],
        "vocabulary": ["collocational precision", "rare but natural lexis", "stylistic variation"],
        "tasks":      ["peer-level intellectual debate", "implicit communication", "full pragmatic competence"],
    },
}

# ── Phoneme curriculum for accent coaching ────────────────────────────────────

PHONEME_UNITS = {
    "american": [
        {
            "unit": 1,
            "name": "TH Sounds",
            "phonemes": ["θ", "ð"],
            "target_words": ["think", "three", "this", "the", "there", "month", "worth"],
            "minimal_pairs": [("think", "sink"), ("three", "free"), ("this", "dis"), ("them", "dem")],
            "practice_sentences": [
                "I think that's the right thing to do.",
                "There are three things I need to tell you.",
                "This is the method I use every month.",
            ],
            "tip": "Put your tongue lightly between your front teeth. For 'th' in 'think' — breathe out (unvoiced). For 'th' in 'this' — vibrate (voiced).",
        },
        {
            "unit": 2,
            "name": "R Sound",
            "phonemes": ["r"],
            "target_words": ["right", "really", "try", "problem", "career", "word"],
            "minimal_pairs": [("rice", "lice"), ("right", "light"), ("red", "led")],
            "practice_sentences": [
                "The right approach really matters in your career.",
                "Try to read three paragraphs out loud every morning.",
            ],
            "tip": "Curl your tongue tip back slightly without touching the roof of your mouth. Don't roll or trill it.",
        },
        {
            "unit": 3,
            "name": "Short Vowels: /æ/ and /ɪ/",
            "phonemes": ["æ", "ɪ"],
            "target_words": ["cat", "bad", "have", "bit", "sit", "fill", "still"],
            "minimal_pairs": [("cat", "cut"), ("bit", "beat"), ("sit", "seat")],
            "practice_sentences": [
                "That bad habit has to stop.",
                "Sit still and fill in the details.",
            ],
            "tip": "For /æ/ — mouth wide open, tongue flat and forward. For /ɪ/ — short and relaxed, NOT the long 'ee'.",
        },
        {
            "unit": 4,
            "name": "V vs W",
            "phonemes": ["v", "w"],
            "target_words": ["very", "have", "voice", "word", "want", "twelve"],
            "minimal_pairs": [("vet", "wet"), ("vine", "wine"), ("vale", "whale")],
            "practice_sentences": [
                "Very well — what do you want to work on?",
                "We have a voice vote every Wednesday.",
            ],
            "tip": "For /v/ — upper teeth touch lower lip, then vibrate. For /w/ — round your lips first, NO teeth contact.",
        },
        {
            "unit": 5,
            "name": "Connected Speech & Rhythm",
            "phonemes": [],
            "target_words": [],
            "minimal_pairs": [],
            "practice_sentences": [
                "I'm going to go and get it.",
                "What do you want to do about it?",
                "I would have done it if I could have.",
            ],
            "tip": "In natural speech, words link together: 'going to' → 'gonna', 'want to' → 'wanna', 'have' → weak /əv/. Rhythm matters more than perfect sounds.",
        },
    ],
    "british": [
        {
            "unit": 1,
            "name": "Non-Rhotic R",
            "phonemes": ["r"],
            "target_words": ["car", "better", "word", "first", "here", "there"],
            "minimal_pairs": [("car/ca", "cart"), ("better", "betta")],
            "practice_sentences": [
                "The first car arrived earlier than we expected.",
                "Better weather here — it's rather nice, actually.",
            ],
            "tip": "In British RP, 'r' after a vowel is usually NOT pronounced (non-rhotic). 'car' sounds like 'caa', 'better' like 'bettuh'.",
        },
        {
            "unit": 2,
            "name": "Short A: /æ/ vs /ɑː/",
            "phonemes": ["æ", "ɑː"],
            "target_words": ["bath", "can't", "dance", "path", "grass", "fast"],
            "minimal_pairs": [("bad", "bard"), ("cat", "cart"), ("Sam", "psalm")],
            "practice_sentences": [
                "She can't dance on the grass path.",
                "After the bath, the class felt calmer.",
            ],
            "tip": "In RP, 'bath', 'can't', 'dance' use the long /ɑː/ (like 'father'), not the short /æ/ of American English.",
        },
        {
            "unit": 3,
            "name": "TH Sounds",
            "phonemes": ["θ", "ð"],
            "target_words": ["think", "three", "this", "the", "together", "although"],
            "minimal_pairs": [("think", "sink"), ("this", "dis"), ("three", "free")],
            "practice_sentences": [
                "I think this approach is thoroughly worthwhile.",
                "Although there are three paths, this one seems right.",
            ],
            "tip": "Tongue lightly between teeth — exactly as in American English. The TH sounds are the same in both accents.",
        },
    ],
}


# ── Session plan builder ──────────────────────────────────────────────────────

def build_session_plan(
    mode: str,
    cefr: str,
    accent: str,
    memory: dict,
    turn_count: int,
    top_errors: list[dict],
    phoneme_unit: Optional[int] = None,
) -> str:
    """
    Generate a curriculum context block to inject into the LLM system prompt.
    This tells the AI exactly what to focus on this turn.
    Returns a formatted string ready to prepend to any system prompt.
    """
    phase = _get_phase(turn_count)
    cefr_upper = cefr.upper() if cefr else "B1"
    skills = CEFR_SKILLS.get(cefr_upper, CEFR_SKILLS["B1"])

    # Pick the primary skill target for this session
    struggles = memory.get("persistent_struggles", [])
    primary_target = _pick_target(mode, cefr_upper, struggles, top_errors)

    # Build phase-specific coaching instruction
    phase_instruction = _phase_instruction(phase, primary_target, cefr_upper)

    # Accent phoneme context
    phoneme_block = ""
    if mode == "accent" and phoneme_unit is not None:
        units = PHONEME_UNITS.get(accent, PHONEME_UNITS["american"])
        unit_idx = min(phoneme_unit, len(units) - 1)
        unit = units[unit_idx]
        phoneme_block = _format_phoneme_unit(unit)

    # Format the full curriculum block
    lines = [
        "━━━ SESSION CURRICULUM (internal — do not quote to user) ━━━",
        f"Phase:          {phase.upper()} (turn {turn_count + 1})",
        f"CEFR Target:    {cefr_upper}",
        f"Primary Focus:  {primary_target}",
        f"Turn Goal:      {phase_instruction}",
    ]

    if skills and mode != "accent":
        if phase == "CORE":
            lines.append(f"Grammar targets for {cefr_upper}: {', '.join(skills['grammar'][:2])}")
        elif phase == "CHALLENGE":
            lines.append(f"Stretch targets: {', '.join(skills['tasks'][:2])}")

    if phoneme_block:
        lines.append(phoneme_block)

    if struggles and phase in ("CORE", "CHALLENGE"):
        lines.append(f"Persistent weaknesses to address (silently): {', '.join(struggles[:3])}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines) + "\n\n"


def build_accent_session_context(
    accent: str,
    phoneme_unit: int,
    turn_count: int,
) -> str:
    """Build a rich accent coaching context block for a specific phoneme unit."""
    phase = _get_phase(turn_count)
    units = PHONEME_UNITS.get(accent, PHONEME_UNITS["american"])
    unit_idx = min(phoneme_unit, len(units) - 1)
    unit = units[unit_idx]

    lines = [
        f"━━━ ACCENT UNIT {unit['unit']}: {unit['name']} ━━━",
        f"Target phonemes: {', '.join(unit['phonemes']) if unit['phonemes'] else 'connected speech patterns'}",
        f"Coaching tip: {unit['tip']}",
        "",
        "TARGET WORDS for this unit: " + ", ".join(unit["target_words"][:5]),
    ]

    if unit["minimal_pairs"]:
        pairs_str = " | ".join(f"{a}/{b}" for a, b in unit["minimal_pairs"][:3])
        lines.append(f"Minimal pairs: {pairs_str}")

    if unit["practice_sentences"]:
        lines.append(f"Practice sentence: \"{unit['practice_sentences'][turn_count % len(unit['practice_sentences'])]}\"")

    lines.append("")
    if phase == "WARMUP":
        lines.append("THIS TURN: Introduce the target sounds with a practice sentence. Model the correct pronunciation clearly.")
    elif phase == "CORE":
        lines.append("THIS TURN: Have the learner repeat a target word/sentence. Give specific, actionable feedback on what they produce.")
    elif phase == "CHALLENGE":
        lines.append("THIS TURN: Give a new spontaneous prompt that naturally requires the target sounds. Do not pre-announce it.")
    else:
        lines.append("THIS TURN: Have a natural mini-conversation that requires the target sounds organically.")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines) + "\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_phase(turn_count: int) -> str:
    if turn_count <= 1:
        return "WARMUP"
    if turn_count <= 4:
        return "CORE"
    if turn_count <= 7:
        return "CHALLENGE"
    return "APPLY"


def _pick_target(mode: str, cefr: str, struggles: list[str], top_errors: list[dict]) -> str:
    """Choose the most relevant skill target for this session."""
    if struggles:
        return struggles[0]

    error_types = [e.get("error_type", "") for e in top_errors if e.get("error_type")]
    if error_types:
        return f"{error_types[0].lower()} accuracy"

    defaults = {
        "training":     f"fluency and {CEFR_SKILLS.get(cefr, {}).get('grammar', ['accuracy'])[0]}",
        "conversation": "natural fluency and vocabulary range",
        "interview":    "professional register and structured responses",
        "professional": "formal vocabulary and coherent argument structure",
        "accent":       "target phoneme production",
        "diagnostic":   "natural speech across all IELTS criteria",
    }
    return defaults.get(mode, "overall fluency and accuracy")


def _phase_instruction(phase: str, target: str, cefr: str) -> str:
    instructions = {
        "WARMUP": f"Build rapport and get the learner talking comfortably. Assess their current {target} level naturally — no pressure.",
        "CORE": f"Actively steer conversation to elicit {target}. Use recasting when errors occur. Push for extended answers.",
        "CHALLENGE": f"Introduce a new angle or hypothetical that requires {target} at slightly above {cefr} level. Notice how they cope.",
        "APPLY": f"Create a spontaneous context where the learner must apply {target} without it being pointed out. Consolidate what was practiced.",
    }
    return instructions.get(phase, instructions["CORE"])


def _format_phoneme_unit(unit: dict) -> str:
    lines = [f"Phoneme focus: {unit['name']} ({', '.join(unit['phonemes']) if unit['phonemes'] else 'prosody'})"]
    lines.append(f"Coaching tip for you to apply: {unit['tip']}")
    if unit.get("practice_sentences"):
        lines.append(f"Use this sentence to model: \"{unit['practice_sentences'][0]}\"")
    return "\n".join(lines)
