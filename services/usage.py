"""
Usage & cost model for the admin panel.

Echo runs entirely on free tiers today (Groq, Google AI Studio, Edge TTS), so
real spend is $0. But to plan for scale, the admin panel shows what the same
traffic *would* cost at each provider's standard published rate. Token counts
are logged per turn in app_events by services/*; this module turns those raw
events into totals, per-stage breakdowns, and an estimated dollar figure.

Rates are USD per 1,000,000 tokens unless noted, and are easy to update in one
place as pricing changes.
"""

from typing import Iterable

# ── Published standard rates (USD). Free tier = $0 actual; these are the
#    "if we paid list price" numbers for capacity planning. ──────────────────
PRICING = {
    # Groq LLMs — per 1M tokens (input / output)
    "openai/gpt-oss-120b":     {"in": 0.15, "out": 0.75},
    "openai/gpt-oss-20b":      {"in": 0.10, "out": 0.50},
    # Legacy Groq Llama ids (kept so old logs still price correctly)
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b-instant":    {"in": 0.05, "out": 0.08},
    # Google Gemini Flash (audio+text) — per 1M tokens (input / output)
    "gemini-flash-latest":     {"in": 0.075, "out": 0.30},
    "gemini-2.5-flash":        {"in": 0.075, "out": 0.30},
    # Fallback for any unrecognised model id
    "_default":                {"in": 0.10, "out": 0.20},
}

# Groq Whisper large-v3 is billed by audio hour (~$0.111/hr).
STT_USD_PER_SECOND = 0.111 / 3600.0

# Microsoft Edge Neural TTS is free in this integration.
TTS_USD_PER_CHAR = 0.0


def _rate(model: str) -> dict:
    return PRICING.get(model or "", PRICING["_default"])


def token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    r = _rate(model)
    return (prompt_tokens / 1_000_000) * r["in"] + (completion_tokens / 1_000_000) * r["out"]


def summarize_events(events: Iterable[dict]) -> dict:
    """
    Aggregate app_events rows (each with a `detail` JSON blob) into a usage &
    cost report: per-stage tokens/cost, per-model tokens, and grand totals.
    """
    stages = {
        "llm":         {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "tts_chars": 0, "cost_usd": 0.0},
        "feedback":    {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        "gemma_audio": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        "stt":         {"calls": 0, "audio_seconds": 0.0, "cost_usd": 0.0},
    }
    per_model: dict = {}

    for e in events:
        et = e.get("event_type")
        d = e.get("detail") or {}
        if not isinstance(d, dict):
            continue

        if et in ("llm", "feedback", "gemma_audio"):
            model = d.get("model") or "_default"
            pt = int(d.get("prompt_tokens") or 0)
            ct = int(d.get("completion_tokens") or 0)
            cost = token_cost(model, pt, ct)
            s = stages[et]
            s["calls"] += 1
            s["prompt_tokens"] += pt
            s["completion_tokens"] += ct
            s["cost_usd"] += cost
            if et == "llm":
                s["tts_chars"] += int(d.get("tts_chars") or 0)
            pm = per_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            pm["prompt_tokens"] += pt
            pm["completion_tokens"] += ct
            pm["cost_usd"] += cost

        elif et == "stt":
            secs = float(d.get("audio_seconds") or 0.0)
            cost = secs * STT_USD_PER_SECOND
            stages["stt"]["calls"] += 1
            stages["stt"]["audio_seconds"] += secs
            stages["stt"]["cost_usd"] += cost

    total_tokens = sum(
        stages[s]["prompt_tokens"] + stages[s]["completion_tokens"]
        for s in ("llm", "feedback", "gemma_audio")
    )
    total_cost = sum(s["cost_usd"] for s in stages.values())

    # Round for presentation
    for s in stages.values():
        s["cost_usd"] = round(s["cost_usd"], 4)
        if "audio_seconds" in s:
            s["audio_seconds"] = round(s["audio_seconds"], 1)
    for m in per_model.values():
        m["cost_usd"] = round(m["cost_usd"], 4)

    return {
        "totals": {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "llm_calls": stages["llm"]["calls"],
            "feedback_calls": stages["feedback"]["calls"],
            "audio_analysis_calls": stages["gemma_audio"]["calls"],
            "stt_calls": stages["stt"]["calls"],
            "stt_audio_seconds": stages["stt"]["audio_seconds"],
            "tts_chars": stages["llm"]["tts_chars"],
        },
        "by_stage": stages,
        "by_model": per_model,
        "note": "Cost is an estimate at each provider's standard list rate; "
                "actual spend on the current free tiers is $0.",
    }
