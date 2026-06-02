"""
LLM service — Google Gemini 2.0 Flash (primary, free 1M tokens/day).
               Groq Llama-3.3-70b (fallback, free 14,400 req/day).

Get your free Gemini key: aistudio.google.com → Get API Key (no card needed)
Get your free Groq key:   console.groq.com → API Keys

Both run in the cloud — no local models, no RAM impact on your laptop.
"""

import json
import os
import time
import logging
from typing import AsyncGenerator

from google import genai as google_genai
from google.genai import types as genai_types
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_FALLBACK = "llama-3.3-70b-versatile"
GROQ_FAST_FB  = "llama-3.1-8b-instant"

# ── Circuit breaker for Gemini quota (429) ────────────────────────────────────
_gemini_quota_until: float = 0.0
GEMINI_BACKOFF_SECS = 300


def _gemini_available() -> bool:
    return time.time() > _gemini_quota_until


def _trip_gemini_circuit(secs: int = GEMINI_BACKOFF_SECS) -> None:
    global _gemini_quota_until
    _gemini_quota_until = time.time() + secs
    logger.warning("[LLM] Gemini quota/error — routing to Groq for %ds", secs)


# ── Clients (lazy init) ────────────────────────────────────────────────────────

_gemini_client: google_genai.Client | None = None
_groq_client: AsyncGroq | None = None


def _get_gemini_client() -> google_genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


async def _gemini_generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 500,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> str | None:
    """
    Single-shot Gemini generation with the new google-genai SDK.
    Returns the text response or None on failure.
    """
    client = _get_gemini_client()
    mime = "application/json" if json_mode else "text/plain"
    cfg = genai_types.GenerateContentConfig(
        system_instruction=system or None,
        max_output_tokens=max_tokens,
        temperature=temperature,
        response_mime_type=mime,
    )
    resp = await client.aio.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=cfg
    )
    return resp.text if resp and resp.text else None


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ── Message format helpers ─────────────────────────────────────────────────────

def _build_gemini_history(history: list[dict]) -> list[dict]:
    """Convert OpenAI-style history to Gemini's {role, parts} format."""
    msgs = []
    for turn in history[-40:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not content or role == "system":
            continue
        gemini_role = "model" if role == "assistant" else "user"
        msgs.append({"role": gemini_role, "parts": [content]})
    return msgs


def _build_groq_messages(system: str, history: list[dict], user_message: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for t in history[-40:]:
        if t.get("role") and t.get("content"):
            msgs.append({"role": t["role"], "content": t["content"]})
    msgs.append({"role": "user", "content": user_message})
    return msgs


# ── Stream B: Conversation ─────────────────────────────────────────────────────

async def stream_conversation(
    system_prompt: str,
    user_message: str,
    state: dict,
) -> AsyncGenerator[str, None]:
    """
    Stream coaching response tokens for a live turn.

    Priority:
      1. Gemini 2.0 Flash — fast, free, cloud
      2. Groq Llama-3.3-70b — free tier fallback
    """
    history = state.get("conversation_history", [])
    past = history[:-1] if history and history[-1].get("role") == "user" else history

    # ── Primary: Gemini Flash (skipped if circuit tripped) ───────────────────
    if _gemini_available():
        try:
            client = _get_gemini_client()
            # Build contents list: system turn + history + current user message
            contents: list = []
            for turn in past[-40:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if not content or role == "system":
                    continue
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(genai_types.Content(role=gemini_role, parts=[genai_types.Part(text=content)]))
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)]))

            cfg = genai_types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                max_output_tokens=180,
                temperature=0.85,
                top_p=0.95,
            )
            got_any = False
            async for chunk in await client.aio.models.generate_content_stream(
                model=GEMINI_MODEL, contents=contents, config=cfg
            ):
                text = chunk.text
                if text:
                    got_any = True
                    yield text
            if got_any:
                return
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                _trip_gemini_circuit()
            # fall through to Groq

    # ── Fallback: Groq Llama-3.3-70b ─────────────────────────────────────────
    try:
        groq = get_groq_client()
        stream = await groq.chat.completions.create(
            model=GROQ_FALLBACK,
            messages=_build_groq_messages(system_prompt, past, user_message),
            stream=True,
            max_tokens=180,
            temperature=0.85,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        return
    except Exception:
        pass

    yield "I missed that — could you say that again?"


# ── Stream C: Correction check ─────────────────────────────────────────────────

async def stream_correction_check(utterance: str) -> dict | None:
    """
    Grammar/error check within a 500ms budget. Never blocks Stream B.
    Uses Groq 8B instant (best for fast structured JSON).
    Falls back to Gemini if Groq quota is exhausted.
    """
    from prompts.correction import CORRECTION_SYSTEM

    clean = _preprocess_utterance(utterance)
    if not clean or len(clean.split()) < 4:
        return None

    # ── Groq 8B instant (fastest JSON) ───────────────────────────────────────
    try:
        groq = get_groq_client()
        response = await groq.chat.completions.create(
            model=GROQ_FAST_FB,
            messages=[
                {"role": "system", "content": CORRECTION_SYSTEM},
                {"role": "user", "content": clean},
            ],
            stream=False,
            max_tokens=220,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, dict) and result.get("error"):
            return result
    except Exception:
        pass

    # ── Gemini fallback (skipped if circuit tripped) ─────────────────────────
    if _gemini_available():
        try:
            resp = await _gemini_generate(
                system=CORRECTION_SYSTEM,
                prompt=f"Check this utterance and respond with ONLY valid JSON:\n\n{clean}",
                max_tokens=220, temperature=0.0, json_mode=True,
            )
            if resp:
                result = json.loads(resp)
                if isinstance(result, dict) and result.get("error"):
                    return result
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                _trip_gemini_circuit()

    return None


# ── Diagnostic scoring ─────────────────────────────────────────────────────────

async def stream_diagnostic_score(transcript: str) -> dict | None:
    """Full IELTS diagnostic scoring — called once after the 5-turn assessment."""
    from prompts.diagnostic import DIAGNOSTIC_SCORING_SYSTEM

    if not transcript or len(transcript.strip()) < 50:
        return None

    scoring_request = (
        f"Full 5-turn speaking assessment transcript:\n\n"
        f"━━━ TRANSCRIPT ━━━\n{transcript}\n━━━ END ━━━\n\n"
        f"Apply IELTS band descriptors precisely. Score each criterion independently. "
        f"Return ONLY valid JSON with no extra text."
    )

    # ── Groq first for scoring (more reliable for large JSON) ────────────────
    # We try Groq Llama-70b first here because diagnostic scoring requires a
    # large, perfectly structured JSON — Groq is more reliable under quota pressure.
    try:
        groq = get_groq_client()
        response = await groq.chat.completions.create(
            model=GROQ_FALLBACK,
            messages=[
                {"role": "system", "content": DIAGNOSTIC_SCORING_SYSTEM},
                {"role": "user", "content": scoring_request},
            ],
            stream=False,
            max_tokens=700,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        validated = _validate_diagnostic(result)
        if validated:
            return validated
    except Exception:
        pass

    # ── Gemini Flash (if Groq failed and circuit not tripped) ────────────────
    if _gemini_available():
        try:
            text = await _gemini_generate(
                prompt=scoring_request, system=DIAGNOSTIC_SCORING_SYSTEM,
                max_tokens=700, temperature=0.1, json_mode=True,
            )
            if text:
                return _validate_diagnostic(json.loads(text))
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                _trip_gemini_circuit()
    return None


# ── Post-session report generation ────────────────────────────────────────────

async def generate_session_report(
    transcript: str,
    phoneme_errors: list[dict],
    cefr_level: str,
) -> dict | None:
    """Generate a full structured session report post-session (slow track)."""
    phoneme_summary = json.dumps(phoneme_errors[:10]) if phoneme_errors else "No phoneme data."

    prompt = f"""You are a professional IELTS/TOEFL analyst. Evaluate this session for a {cefr_level} learner.

TRANSCRIPT:
{transcript[:4000]}

PHONETIC ANALYSIS (IPA mismatch data from Wav2Vec2):
{phoneme_summary}

Return ONLY a valid JSON object — no markdown, no explanation, just JSON:
{{
  "ielts_estimated_band": <float 1.0-9.0>,
  "grammar_mistakes": [
    {{"original": "...", "correction": "...", "rule": "...", "severity": "minor|major"}}
  ],
  "vocabulary_upgrades": [
    {{"used": "...", "better_alternative": "...", "register": "formal|informal|neutral"}}
  ],
  "accent_coaching": [
    {{"target_word": "...", "detected_ipa": "...", "target_ipa": "...", "coaching_tip": "..."}}
  ],
  "fluency_notes": "...",
  "strengths": ["..."],
  "next_session_focus": ["..."]
}}

Max 5 items per list. Be specific and actionable."""

    # ── Groq first (more reliable under Gemini quota pressure) ───────────────
    try:
        groq = get_groq_client()
        response = await groq.chat.completions.create(
            model=GROQ_FALLBACK,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=900,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        pass

    # ── Gemini fallback ───────────────────────────────────────────────────────
    if _gemini_available():
        try:
            text = await _gemini_generate(prompt=prompt, max_tokens=900, temperature=0.1, json_mode=True)
            if text:
                return json.loads(text)
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                _trip_gemini_circuit()
    return None


# ── Memory insight extraction ──────────────────────────────────────────────────

async def extract_session_insights(transcript: str, top_errors: list) -> dict | None:
    """Extract structured learning insights post-session."""
    prompt = f"""Analyse this English coaching session and extract learning insights.

TRANSCRIPT:
{transcript[:3000]}

TOP ERRORS: {json.dumps([e.get('error_type', '') for e in top_errors[:5]])}

Return ONLY valid JSON:
{{
  "mastered": ["..."],
  "persistent_struggles": ["..."],
  "preferred_topics": ["..."],
  "personality_notes": "...",
  "last_session_summary": "..."
}}

Be specific (e.g. "subject-verb agreement", not "grammar"). Max 3 items per list."""

    # ── Groq first ────────────────────────────────────────────────────────────
    try:
        groq = get_groq_client()
        resp = await groq.chat.completions.create(
            model=GROQ_FAST_FB,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=320,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _preprocess_utterance(text: str) -> str:
    text = text.strip()
    for filler in ["um,", "uh,", "hmm,", "er,"]:
        if text.lower().startswith(filler):
            text = text[len(filler):].strip()
    return text


_VALID_MODES = {"training", "conversation", "interview", "professional", "accent"}


def _validate_diagnostic(result: dict) -> dict | None:
    required = ["cefr_level", "ielts_equivalent", "fluency", "lexical", "grammar", "pronunciation"]
    if not all(k in result for k in required):
        return None
    for field in ["ielts_equivalent", "fluency", "lexical", "grammar", "pronunciation"]:
        if field in result:
            try:
                result[field] = max(0.0, min(9.0, float(result[field])))
            except (TypeError, ValueError):
                result[field] = 0.0

    result["roadmap"] = _normalize_roadmap(result)
    return result


def _normalize_roadmap(result: dict) -> dict:
    """
    Clean the AI-generated roadmap into a stable shape the app can rely on:
      { goal, motivation, timeline, completed_count, sessions: [ {n, title, mode, accent,
        focus, description, skills[], duration_min, status} ] }
    Falls back to a sensible default plan if the model returned nothing usable.
    """
    raw = result.get("roadmap") or {}
    sessions_in = raw.get("sessions") if isinstance(raw, dict) else None
    if not isinstance(sessions_in, list):
        sessions_in = []

    goal       = (result.get("goal") or raw.get("goal") or "General English fluency").strip()
    motivation = (result.get("motivation") or raw.get("motivation") or "").strip()
    timeline   = (result.get("timeline") or raw.get("timeline") or "flexible").strip()

    clean: list[dict] = []
    for i, s in enumerate(sessions_in):
        if not isinstance(s, dict):
            continue
        mode = str(s.get("mode", "training")).lower().strip()
        if mode not in _VALID_MODES:
            mode = "training"
        title = str(s.get("title") or f"Session {i + 1}").strip()
        skills = s.get("skills") if isinstance(s.get("skills"), list) else []
        try:
            duration = int(s.get("duration_min", 10))
        except (TypeError, ValueError):
            duration = 10
        clean.append({
            "n": i + 1,
            "title": title[:80],
            "mode": mode,
            "accent": str(s.get("accent", "american")).lower().strip() or "american",
            "focus": str(s.get("focus", "")).strip()[:80],
            "description": str(s.get("description", "")).strip()[:280],
            "skills": [str(x).strip()[:40] for x in skills][:5],
            "duration_min": max(5, min(20, duration)),
            "status": "pending",
        })

    # Fallback plan if the model returned an empty / malformed roadmap
    if not clean:
        clean = [
            {"n": 1, "title": "Fluency & Accuracy Warm-up", "mode": "training", "accent": "american",
             "focus": result.get("focus_area", "core accuracy"), "description": "Targeted coaching on your most common errors through natural conversation.",
             "skills": [], "duration_min": 10, "status": "pending"},
            {"n": 2, "title": "Build Natural Fluency", "mode": "conversation", "accent": "american",
             "focus": "fluency", "description": "Free-flowing dialogue to build rhythm and confidence.",
             "skills": [], "duration_min": 12, "status": "pending"},
            {"n": 3, "title": "Vocabulary & Range", "mode": "training", "accent": "american",
             "focus": "lexical resource", "description": "Upgrade everyday words to precise, higher-band alternatives.",
             "skills": [], "duration_min": 10, "status": "pending"},
        ]

    return {
        "goal": goal,
        "motivation": motivation,
        "timeline": timeline,
        "completed_count": 0,
        "sessions": clean,
    }
