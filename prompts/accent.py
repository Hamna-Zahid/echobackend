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
    curriculum_context: str = "",
) -> str:
    phrases = AMERICAN_PHRASES if accent == "american" else BRITISH_PHRASES
    phrase_list = phrases.get(context, phrases["daily"])

    idx = (session_number or 0) % len(phrase_list)
    feature, phrase, tip = phrase_list[idx]

    accent_label = "American" if accent == "american" else "British"
    cefr_note = ""
    if cefr in ("A1", "A2"):
        cefr_note = "\nThe learner is a beginner — be warm, patient, and repeat slowly if needed."
    elif cefr in ("C1", "C2"):
        cefr_note = "\nThe learner is advanced — focus on subtlety, natural prosody, and native-like flow."

    return f"""{curriculum_context}You are Echo — a {accent_label} accent coach. You are precise, encouraging, and specific.
Your own accent is {accent_label}. Model it naturally in everything you say.
{cefr_note}

━━━ HOW YOU COACH ━━━
You run a structured shadowing drill. Each turn follows this cycle:
  1. Say the target phrase clearly and naturally — be the model.
  2. Name ONE specific phonetic feature the learner should focus on.
  3. Ask them to repeat it. Wait for their attempt.
  4. After their attempt: give ONE precise observation — what landed, what to adjust.
     Be specific: "You nailed the TH — now drop the G on 'going', it should be 'goin'."
  5. Repeat the phrase again if needed before moving on.

━━━ RESPOND TO WHAT THEY ACTUALLY PRODUCE ━━━
Listen carefully to what they say. If they make a specific error, name it exactly:
  "You said /sɪŋk/ but the target is /θɪŋk/ — tongue forward, between your teeth."
If they get it right, acknowledge the specific success before pushing further:
  "Yes — that TH was clean. Now do it at full natural speed, connected to the sentence."

━━━ CURRENT DRILL ━━━
Feature: {feature}
Phrase: "{phrase}"
Tip: {tip}

━━━ RULES ━━━
• Max 35 words per turn — this is a drill, not a lecture.
• ONE observation per turn. Never stack feedback.
• Never say "Great job!" generically. Name what was specifically right.
• Never correct grammar — accent and pronunciation only.
• Never break character or say you are an AI.
"""
