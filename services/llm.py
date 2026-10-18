"""
LLM service — multi-provider (Groq default, Cerebras, OpenRouter).

Providers are OpenAI-compatible chat-completions endpoints:

  groq        → openai/gpt-oss-20b (fast) / openai/gpt-oss-120b (feedback)
  cerebras    → qwen-3-32b / llama-3.3-70b (very low TTFT)
  openrouter  → route to the fastest available model per prompt

Select with LLM_PROVIDER. Conversation can independently prefer Cerebras via
LLM_CONVERSATION_PREFER_CEREBRAS=true while feedback stays on Groq.

Model defaults (env-overridable):
  Conversation → openai/gpt-oss-20b   (low-latency, spoken)
  Feedback     → openai/gpt-oss-120b  (JSON mode, higher quality; short
                                       payloads drop to the conversation model)
  Utilities    → openai/gpt-oss-20b   (cheap + instant)
"""

import json
import os
from typing import AsyncGenerator, Optional

import httpx
from groq import AsyncGroq

_client: AsyncGroq | None = None

# ── Groq model defaults ───────────────────────────────────────────────────────
CONVERSATION_MODEL = os.environ.get("CONVERSATION_MODEL", "openai/gpt-oss-20b")
FEEDBACK_MODEL     = os.environ.get("FEEDBACK_MODEL", "openai/gpt-oss-120b")
FAST_MODEL         = os.environ.get("FAST_MODEL", "openai/gpt-oss-20b")

# ── Provider selection ────────────────────────────────────────────────────────
PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
PREFER_CEREBRAS = os.environ.get(
    "LLM_CONVERSATION_PREFER_CEREBRAS", "false"
).lower() in ("1", "true", "yes", "on")

# Cerebras (OpenAI-compatible)
CEREBRAS_BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_CONV     = os.environ.get("CEREBRAS_CONVERSATION_MODEL", "qwen-3-32b")
CEREBRAS_FEED     = os.environ.get("CEREBRAS_FEEDBACK_MODEL", "llama-3.3-70b")
CEREBRAS_FAST     = os.environ.get("CEREBRAS_FAST_MODEL", "qwen-3-8b")

# OpenRouter (OpenAI-compatible)
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_CONV     = os.environ.get(
    "OPENROUTER_CONVERSATION_MODEL", "qwen/qwen-2.5-7b-instruct")
OPENROUTER_FEED     = os.environ.get(
    "OPENROUTER_FEEDBACK_MODEL", "openai/gpt-oss-120b")
OPENROUTER_FAST     = os.environ.get(
    "OPENROUTER_FAST_MODEL", "qwen/qwen-2.5-7b-instruct")


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = AsyncGroq(api_key=api_key)
    return _client


# ── Provider/model resolution ─────────────────────────────────────────────────

def _use_cerebras_conv() -> bool:
    return PROVIDER == "cerebras" or PREFER_CEREBRAS


def _choose_conversation_model() -> str:
    if _use_cerebras_conv():
        return CEREBRAS_CONV
    if PROVIDER == "openrouter":
        return OPENROUTER_CONV
    return CONVERSATION_MODEL


def _choose_feedback_model(payload: str = "") -> str:
    short = False
    try:
        short = len(payload or "") < 1200
    except Exception:
        short = False
    if PROVIDER == "cerebras":
        return CEREBRAS_FAST if short else CEREBRAS_FEED
    if PROVIDER == "openrouter":
        return OPENROUTER_FAST if short else OPENROUTER_FEED
    # Groq: cheap model for short scorecards, high-quality 120b otherwise.
    return CONVERSATION_MODEL if short else FEEDBACK_MODEL


def _choose_fast_model() -> str:
    if PROVIDER == "cerebras":
        return CEREBRAS_FAST
    if PROVIDER == "openrouter":
        return OPENROUTER_FAST
    return FAST_MODEL


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

    If `usage_sink` is given, token usage from the final stream chunk is written
    into it for the admin usage/cost panel.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-40:]:
        if turn.get("role") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    if user_message and (not history or history[-1].get("role") != "user"):
        messages.append({"role": "user", "content": user_message})

    model = _choose_conversation_model()

    if PROVIDER in ("cerebras", "openrouter") or PREFER_CEREBRAS:
        base, key = (
            (CEREBRAS_BASE_URL, CEREBRAS_API_KEY)
            if _use_cerebras_conv()
            else (OPENROUTER_BASE_URL, OPENROUTER_API_KEY)
        )
        if not key:
            # No key for the preferred provider → fall back to Groq so the
            # session keeps speaking.
            pass
        else:
            produced = False
            async for token in _openai_stream(base, key, model, messages, usage_sink):
                produced = True
                yield token
            if produced:
                return

    # Groq (default / fallback).
    try:
        client = get_groq_client()
        kwargs = dict(
            model=model, messages=messages, stream=True,
            max_tokens=180, temperature=0.7, top_p=0.9,
        )
        # reasoning_effort is only valid on the gpt-oss family.
        if isinstance(model, str) and model.startswith("openai/gpt-oss-"):
            kwargs["reasoning_effort"] = "low"
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            _capture_stream_usage(chunk, usage_sink, model)
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


# ── OpenAI-compatible streaming (Cerebras / OpenRouter) ───────────────────────

async def _openai_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    usage_sink: dict | None = None,
    max_tokens: int = 180,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> AsyncGenerator[str, None]:
    if not api_key:
        return
    payload = {
        "model": model, "messages": messages, "stream": True,
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            async with client.stream(
                "POST", f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            ) as response:
                if response.status_code != 200:
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str in ("", "[DONE]"):
                        continue
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
    except Exception:
        return


async def _openai_json(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> dict | None:
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_payload = {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    for use_json in (True, False):
        payload = dict(base_payload)
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                continue
            content = resp.json()["choices"][0]["message"]["content"]
            if not content:
                continue
            if use_json:
                return json.loads(content)
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start:end + 1])
        except Exception:
            continue
    return None


# ── Live grammar correction (Stream C) ────────────────────────────────────────

async def run_correction_check(utterance: str) -> dict | None:
    """Live grammar correction on the user's words. Returns a card dict or None."""
    clean = (utterance or "").strip()
    if len(clean.split()) < 4:
        return None

    try:
        from prompts.correction import CORRECTION_SYSTEM
    except Exception:
        CORRECTION_SYSTEM = (
            "You are a precise spoken-English corrector. Return ONLY JSON with "
            "keys error (bool), original, corrected, explanation, and error_type."
        )

    messages = [
        {"role": "system", "content": CORRECTION_SYSTEM},
        {"role": "user", "content": clean},
    ]

    if PROVIDER == "cerebras":
        result = await _openai_json(CEREBRAS_BASE_URL, CEREBRAS_API_KEY,
                                    CEREBRAS_FAST, messages, max_tokens=220, temperature=0.0)
    elif PROVIDER == "openrouter":
        result = await _openai_json(OPENROUTER_BASE_URL, OPENROUTER_API_KEY,
                                    OPENROUTER_FAST, messages, max_tokens=220, temperature=0.0)
    else:
        try:
            client = get_groq_client()
            response = await client.chat.completions.create(
                model=FAST_MODEL,
                messages=messages,
                stream=False,
                max_tokens=220,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        except Exception:
            return None

    if not isinstance(result, dict) or not result.get("error"):
        return None
    if not result.get("original") or not result.get("corrected"):
        return None
    return result


# ── Feedback engine: structured scoring (the part people pay for) ─────────────

async def generate_feedback_json(
    system_prompt: str,
    user_payload: str,
    usage_sink: dict | None = None,
) -> dict | None:
    """
    Produce the structured coaching report in JSON mode. Model is chosen per
    provider and payload size (short scorecards use the cheaper model).
    Token usage (if `usage_sink` is given) is captured for the admin panel.
    """
    model = _choose_feedback_model(user_payload)

    if PROVIDER == "cerebras":
        return await _openai_json(CEREBRAS_BASE_URL, CEREBRAS_API_KEY, model,
                                  [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_payload}],
                                  max_tokens=2000, temperature=0.2)
    if PROVIDER == "openrouter":
        return await _openai_json(OPENROUTER_BASE_URL, OPENROUTER_API_KEY,
                                  model,
                                  [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_payload}],
                                  max_tokens=2000, temperature=0.2)

    try:
        client = get_groq_client()
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            stream=False,
            max_tokens=2000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        if isinstance(model, str) and model.startswith("openai/gpt-oss-"):
            kwargs["reasoning_effort"] = "low"
        response = await client.chat.completions.create(**kwargs)
        _capture_response_usage(response, usage_sink, model)
        return json.loads(response.choices[0].message.content)
    except Exception:
        # Retry once without JSON mode, then best-effort parse.
        try:
            client = get_groq_client()
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nReturn ONLY valid JSON."},
                    {"role": "user", "content": user_payload},
                ],
                stream=False,
                max_tokens=2000,
                temperature=0.2,
            )
            if isinstance(model, str) and model.startswith("openai/gpt-oss-"):
                kwargs["reasoning_effort"] = "low"
            response = await client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
        except Exception:
            pass
        return None
