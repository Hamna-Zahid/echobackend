"""
Accent training prompt builder.

Shadowing drill: Echo says a phrase, user repeats it, Echo gives brief coaching.
Focuses on specific phonetic features per accent and context.
"""

from typing import Optional, Literal


# Phrases grouped by context, each with the key feature to coach
AMERICAN_PHRASES = {
    "daily": [
        ("T-flapping",  "I need to think about it.",            "The 'T' in 'need to' sounds like a soft D — 'nee-duh think about it.'"),
        ("Rhotic R",    "I heard her car start early.",          "Strong R on 'her', 'car', 'early' — don't drop the R."),
        ("Schwa",       "I can probably figure it out.",         "Unstressed vowels become schwa — 'prəbəbly', 'figyer it out'."),
        ("Linking",     "Come on in and sit down.",              "Link words: 'Come-on-in', 'si-down' — no pauses between them."),
    ],
    "ielts": [
        ("Stress",      "The most significant advantage is...",  "Stress 'SIG-ni-fi-cant', 'ad-VAN-tage' — content words carry the stress."),
        ("Reduction",   "As a result of the changes,",           "'Aza result' — 'as a' reduces to one smooth unit."),
        ("Intonation",  "On the other hand, it could be argued", "Rise-fall on key phrases. 'On the OTHER hand' — OTHER gets the peak."),
    ],
    "toefl": [
        ("Prominence",  "Studies have shown that...",            "Stress 'STUD-ies have SHOWN' — verb and noun get prominence."),
        ("Rhythm",      "In addition to the benefits,",          "English is stress-timed: 'in-ADD-ition to the BEN-efits' — unstressed syllables compress."),
    ],
    "interview": [
        ("Confidence",  "I led the team through a difficult transition.", "Steady pace, slight emphasis on 'LED' and 'difficult' — avoid trailing off."),
        ("Clarity",     "My key contribution was improving efficiency.",   "Clear consonants on 'key', 'contribution', 'improving' — don't swallow endings."),
    ],
}

BRITISH_PHRASES = {
    "daily": [
        ("Non-rhotic R", "I parked the car in the garden.",      "Don't pronounce the R in 'park', 'car', 'garden' — 'pahk', 'cah', 'gahden'."),
        ("T clarity",    "Put the kettle on, will you?",         "British T is crisp: 'puT', 'keTTle' — not flapped like American English."),
        ("Broad A",      "I can't dance in the afternoon.",      "'Can't' = /kɑːnt/, 'dance' = /dɑːns/, 'after' = /ˈɑːftə/ — long open vowel."),
        ("Linking",      "Turn it off and put it away.",         "Smooth linking: 'Turn-it-off', 'pu-ti-taway' — natural rhythm, no gaps."),
    ],
    "ielts": [
        ("Hedging",     "It could be argued that...",            "Soft falling intonation: 'It COULD be ARgued' — measured, academic tone."),
        ("Discourse",   "Furthermore, the evidence suggests...", "Clear stress on discourse markers: 'FURthermore', 'EVidence' — they signal structure."),
    ],
    "toefl": [
        ("Pace",        "Research indicates a direct correlation.", "Steady pace, clear syllables: 're-SEARCH in-DI-cates' — don't rush the long words."),
    ],
    "interview": [
        ("Register",    "I'd be keen to take that forward.",     "Formal British: 'keen', 'take that forward' — confident but understated tone."),
        ("Precision",   "I ensured the project stayed on track.", "British interview register: precise consonants, measured delivery."),
    ],
}


def build_accent_prompt(
    accent: str,
    cefr: str,
    context: str = "daily",
    session_number: Optional[int] = None,
) -> str:
    phrases = AMERICAN_PHRASES if accent == "american" else BRITISH_PHRASES
    phrase_list = phrases.get(context, phrases["daily"])

    # Pick phrase based on session number for variety
    idx = (session_number or 0) % len(phrase_list)
    feature, phrase, tip = phrase_list[idx]

    accent_label = "American" if accent == "american" else "British"
    cefr_note = ""
    if cefr in ("A1", "A2"):
        cefr_note = "\nThe learner is a beginner — use encouragement and repeat the phrase slowly if needed."
    elif cefr in ("C1", "C2"):
        cefr_note = "\nThe learner is advanced — focus on subtlety and natural flow, not just correct pronunciation."

    return f"""You are Echo — a {accent_label} accent coach running a shadowing drill.
Your accent is {accent_label}. Model it precisely in everything you say.
{cefr_note}

━━━ SESSION STRUCTURE ━━━
1. Say a phrase clearly and naturally.
2. Tell the learner ONE specific feature to listen for.
3. Ask them to repeat the phrase.
4. After they repeat: give ONE specific piece of feedback (what was good, what to adjust).
5. Repeat the phrase again if needed, then move to the next one.

━━━ CURRENT DRILL ━━━
Feature to focus on: {feature}
Phrase: "{phrase}"
Coaching note: {tip}

Start with: "Listen carefully — I'll say this phrase: '{phrase}'.
Focus on the {feature}. {tip}
Now you repeat it — go ahead."

━━━ COACHING RULES ━━━
• Keep each response under 30 words.
• Praise what was correct specifically: "Good — you got the R. Now try the ending more clearly."
• Never say "Great job!" generically — name the specific thing they did right.
• Never correct grammar. Accent only.
• Never reveal you are an AI.
"""
