"""
Offline test suite — validates the cascade's pure logic and the FREE Edge TTS
without needing any API keys or network to Groq/Supabase.

Run:  python test_offline.py
(Groq/LLM calls are mocked; Edge TTS is exercised for real — it's free.)
"""

import asyncio
import os
import sys

# Dummy env so modules that read os.environ at import don't blow up.
os.environ.setdefault("GROQ_API_KEY", "test-key")
# Force the Gemma sidecar into its no-key degradation path for offline testing.
os.environ.pop("GEMINI_API_KEY", None)

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


async def main():
    print("\n== STT filler counting ==")
    from services.stt import count_fillers
    hits = count_fillers("So um, I basically, you know, like did the thing, um.")
    check("counts 'um' twice", hits.get("um") == 2)
    check("counts 'like'", hits.get("like") == 1)
    check("counts 'you know'", hits.get("you know") == 1)
    check("no false positives on clean text", count_fillers("I led the migration and shipped it.") == {})

    print("\n== Feedback metrics ==")
    from services.feedback import compute_delivery_metrics, _format_transcript, generate_session_feedback
    m = compute_delivery_metrics(total_user_words=150, total_user_seconds=60.0,
                                 filler_hits={"um": 5, "like": 3})
    check("wpm computed (150 words / 1 min = 150)", m["pace_wpm"] == 150)
    check("filler_count summed", m["filler_count"] == 8)
    check("top_fillers ranked", m["top_fillers"][0] == "um")

    turns = [{"role": "assistant", "content": "Tell me about a challenge."},
             {"role": "user", "content": "I handled a tough migration."}]
    tx = _format_transcript(turns, "interview")
    check("transcript labels interviewer/candidate", "Interviewer:" in tx and "Candidate:" in tx)

    print("\n== Feedback generation (mocked LLM) ==")
    import services.feedback as fb
    async def fake_json(system, payload):
        return {"overall_score": 72, "headline": "Solid, tighten structure",
                "delivery": {"pace_wpm": 999, "filler_count": 999}, "strengths": ["clear"]}
    fb.generate_feedback_json = fake_json
    report = await generate_session_feedback("interview", turns * 5, m)  # enough words
    check("report returned", report is not None)
    check("measured wpm overrides LLM guess", report["delivery"]["pace_wpm"] == 150)
    check("measured filler_count overrides LLM guess", report["delivery"]["filler_count"] == 8)

    short = await generate_session_feedback("interview", turns, {"pace_wpm": 0, "filler_count": 0,
                                            "top_fillers": [], "spoken_seconds": 1, "word_count": 5})
    check("too-short session returns no report", short is None)

    print("\n== Gemma 4 audio module (pure logic) ==")
    from services.gemma_audio import _sniff_mime, format_delivery_notes, analyze_delivery
    check("sniffs wav", _sniff_mime(b"RIFF" + b"\x00" * 20) == "audio/wav")
    check("sniffs webm", _sniff_mime(b"\x1aE\xdf\xa3" + b"\x00" * 20) == "audio/webm")
    notes_txt = format_delivery_notes([{"confidence": 7, "tone": "warm", "hesitancy": "low",
                                        "pronunciation_flags": ["specifically"]}])
    check("delivery notes formatted", "confidence=7/10" in notes_txt and "specifically" in notes_txt)
    check("empty notes -> empty string", format_delivery_notes([]) == "")
    no_key = await analyze_delivery(b"RIFF" + b"\x00" * 500, 10.0)  # GEMINI_API_KEY unset
    check("no key -> graceful None (degrades)", no_key is None)

    print("\n== Edge TTS (FREE, real) ==")
    try:
        from services.tts import stream_tts
        chunks = []
        async for c in stream_tts("Hello, this is a test of the coaching voice.", accent="american"):
            chunks.append(c)
        total = sum(len(c) for c in chunks)
        check("edge TTS produced audio bytes", total > 1000)
    except Exception as e:
        check(f"edge TTS ran (got: {e})", False)

    print(f"\n==== {passed} passed, {failed} failed ====")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
