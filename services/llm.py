"""
LLM service — Groq only (FREE tier). No OpenAI, no paid APIs.

Groq serves open-source Llama models on its own fast inference hardware, so you
get open-source models *and* the fastest first-token latency available, for $0
on the free tier.

  Conversation (interviewer / partner turns) → llama-3.3-70b-versatile (smart, fast)
  Feedback / scoring (the product's moat)     → llama-3.3-70b-versatile (JSON mode)
  Quick utilities                             → llama-3.1-8b-instant   (cheap + instant)
"""

import json
import os
from typing import AsyncGenerator

from groq import AsyncGroq

_client: AsyncGroq | None = None

CONVERSATION_MODEL = "llama-3.3-70b-versatile"
FEEDBACK_MODEL     = "llama-3.3-70b-versatile"
FAST_MODEL         = "llama-3.1-8b-instant"


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


# ── Live conversation: interviewer / speech partner turns ─────────────────────

async def stream_conversation(
    system_prompt: str,
    history: list[dict],
    user_message: str | None = None,
    usage_sink: dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the AI's next spoken turn (interviewer question, follow-up, or partner
    reply). `history` is the running list of {role, content} turns.

    If `usage_sink` is given, the Groq token usage (prompt/completion/total) from
    the final stream chunk is written into it for the admin usage/cost panel.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-40:]:
        if turn.get("role") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    if user_message and (not history or history[-1].get("role") != "user"):
        messages.append({"role": "user", "content": user_message})

    try:
        client = get_groq_client()
        stream = await client.chat.completions.create(
            model=CONVERSATION_MODEL,
            messages=messages,
            stream=True,
            max_tokens=180,
            temperature=0.7,
            top_p=0.9,
        )
        async for chunk in stream:
            _capture_stream_usage(chunk, usage_sink, CONVERSATION_MODEL)
            # The final usage chunk can carry an empty choices list — guard it.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception:
        yield "Sorry, could you repeat that?"


def _capture_stream_usage(chunk, sink: dict | None, model: str) -> None:
    """Groq streams token usage on the final chunk under `x_groq.usage`."""
    if sink is None:
        return
    xg = getattr(chunk, "x_groq", None)
    usage = getattr(xg, "usage", None) if xg is not None else None
    if usage is None:
        return
    sink["model"] = model
    sink["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
    sink["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
    sink["total_tokens"] = int(getattr(usage, "total_tokens", 0) or 0)


def _capture_response_usage(response, sink: dict | None, model: str) -> None:
    """Non-streaming Groq calls expose usage directly on the response."""
    if sink is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    sink["model"] = model
    sink["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
    sink["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
    sink["total_tokens"] = int(getattr(usage, "total_tokens", 0) or 0)


# ── Stream C: live grammar correction check ──────────────────────────────────

async def run_correction_check(utterance: str) -> dict | None:
    """
    Fast grammar/error check on one user utterance (Groq 8B instant, JSON mode).
    Returns the correction dict when a real error is found, else None.
    Fire-and-forget from the session loop — must never block or crash a turn.
    """
    from prompts.correction import CORRECTION_SYSTEM

    clean = utterance.strip()
    if len(clean.split()) < 4:
        return None  # too short to judge fairly

    try:
        client = get_groq_client()
        response = await client.chat.completions.create(
            model=FAST_MODEL,
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
        if not isinstance(result, dict) or not result.get("error"):
            return None
        # Require the schema fields the app renders.
        if not result.get("original") or not result.get("corrected"):
            return None
        return result
    except Exception:
        return None


# ── Feedback engine: structured scoring (the part people pay for) ─────────────

async def generate_feedback_json(
    system_prompt: str,
    user_payload: str,
    usage_sink: dict | None = None,
) -> dict | None:
    """
    Call Llama 3.3 70B in JSON mode to produce the structured coaching report.
    Returns a dict, or None on failure. Token usage (if `usage_sink` is given)
    is captured for the admin usage/cost panel.
    """
    try:
        client = get_groq_client()
        response = await client.chat.completions.create(
            model=FEEDBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            stream=False,
            max_tokens=2000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        _capture_response_usage(response, usage_sink, FEEDBACK_MODEL)
        return json.loads(response.choices[0].message.content)
    except Exception:
        # Retry once without JSON mode, then best-effort parse.
        try:
            client = get_groq_client()
            response = await client.chat.completions.create(
                model=FEEDBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nReturn ONLY valid JSON."},
                    {"role": "user", "content": user_payload},
                ],
                stream=False,
                max_tokens=2000,
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
        except Exception:
            pass
        return None
