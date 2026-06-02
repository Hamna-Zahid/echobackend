"""
Steps 3–7 pipeline verification.
Tests each service individually and then the full chain.

Usage:
  python test_pipeline.py          # run all tests
  python test_pipeline.py stt      # test STT only
  python test_pipeline.py llm      # test LLM only
  python test_pipeline.py tts      # test TTS only
  python test_pipeline.py full     # full pipeline

Requires .env with GROQ_API_KEY and CARTESIA_API_KEY set.
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()


# ── Step 3: Groq Whisper STT ──────────────────────────────────────────────

async def test_stt():
    """Generate a minimal WAV file and transcribe it."""
    from services.stt import transcribe_audio
    import struct, math

    # Generate 2s of 440Hz sine wave as WAV (simulates speech for API test)
    sample_rate = 16000
    duration = 2
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(sample_rate * duration)]

    # Build WAV bytes
    data = struct.pack(f"<{len(samples)}h", *samples)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(data), b'WAVE', b'fmt ', 16,
        1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(data))
    wav = header + data

    t0 = time.perf_counter()
    result = await transcribe_audio(wav)
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"[Step 3] STT: '{result}' — {ms}ms")
    assert ms < 2000, f"STT latency {ms}ms exceeds 2s budget"
    return result


# ── Step 4: Groq LLM streaming ────────────────────────────────────────────

async def test_llm():
    from services.llm import stream_conversation
    from prompts.training import build_training_prompt

    prompt = build_training_prompt("B2", [], "technology", "american")
    state = {"conversation_history": [], "cefr_level": "B2"}

    tokens = []
    t0 = time.perf_counter()
    first_token_ms = None

    async for token in stream_conversation(prompt, "Tell me about AI in education.", state):
        if first_token_ms is None:
            first_token_ms = int((time.perf_counter() - t0) * 1000)
        tokens.append(token)

    total = "".join(tokens)
    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[Step 4] LLM first token: {first_token_ms}ms | total: {total_ms}ms")
    print(f"         Response: {total[:120]}...")
    assert first_token_ms < 500, f"First token {first_token_ms}ms exceeds 500ms budget"
    return total


# ── Step 5: Cartesia TTS streaming ───────────────────────────────────────

async def test_tts():
    from services.tts import stream_tts

    text = "Hello, I am your English coach. Let's begin with a quick question."
    chunks = []
    t0 = time.perf_counter()
    first_chunk_ms = None

    async for chunk in stream_tts(text, accent="american"):
        if chunk and first_chunk_ms is None:
            first_chunk_ms = int((time.perf_counter() - t0) * 1000)
        if chunk:
            chunks.append(chunk)

    total_bytes = sum(len(c) for c in chunks)
    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[Step 5] TTS first chunk: {first_chunk_ms}ms | total: {total_ms}ms | {total_bytes} bytes")
    assert first_chunk_ms is None or first_chunk_ms < 1000, f"TTS first chunk {first_chunk_ms}ms too slow"
    return chunks


# ── Step 6/7: Full pipeline (STT → LLM → TTS) ────────────────────────────

async def test_full_pipeline():
    import struct, math

    print("[Step 6] Full pipeline: audio → transcript → LLM → TTS audio")

    # Synthetic audio
    sr = 16000
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(sr * 2)]
    data = struct.pack(f"<{len(samples)}h", *samples)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36+len(data), b'WAVE', b'fmt ', 16,
        1, 1, sr, sr*2, 2, 16, b'data', len(data))

    t_start = time.perf_counter()

    from services.stt import transcribe_audio
    from services.llm import stream_conversation
    from services.tts import stream_tts
    from prompts.training import build_training_prompt

    # STT
    t0 = time.perf_counter()
    transcript = await transcribe_audio(header + data)
    t_stt = int((time.perf_counter() - t0) * 1000)

    # LLM (measure first token)
    prompt = build_training_prompt("B2", [], "general", "american")
    state = {"conversation_history": [], "cefr_level": "B2"}

    tokens = []
    t0 = time.perf_counter()
    first_token_ms = None

    async for tok in stream_conversation(prompt, transcript or "Hello", state):
        if first_token_ms is None:
            first_token_ms = int((time.perf_counter() - t0) * 1000)
        tokens.append(tok)
        # Simulate sentence boundary after first chunk for latency test
        if len("".join(tokens)) > 40 and "".join(tokens).rstrip()[-1] in ".!?":
            break

    first_sentence = "".join(tokens).strip()

    # TTS (measure first audio chunk)
    t0 = time.perf_counter()
    first_audio_ms = None
    async for chunk in stream_tts(first_sentence, accent="american"):
        if chunk and first_audio_ms is None:
            first_audio_ms = int((time.perf_counter() - t0) * 1000)
        break  # just measure first chunk

    total_ms = int((time.perf_counter() - t_start) * 1000)

    print(f"\n  STT:              {t_stt}ms      (target < 200ms)")
    print(f"  LLM first token:  {first_token_ms}ms   (target < 200ms)")
    print(f"  TTS first chunk:  {first_audio_ms}ms  (target < 300ms)")
    print(f"  Total E2E:        {total_ms}ms   (target < 750ms)")
    print(f"  Transcript:       '{transcript}'")
    print(f"  First sentence:   '{first_sentence[:80]}'")

    if total_ms < 750:
        print("\n  [PASS] Sub-750ms target achieved!")
    else:
        print(f"\n  [WARN] {total_ms}ms exceeds 750ms target — check network/API region")


# ── Entry ────────────────────────────────────────────────────────────────

async def main(test: str = "all"):
    print("EchoCoach Pipeline Tests\n" + "─" * 40)

    if not os.environ.get("GROQ_API_KEY"):
        print("[SKIP] GROQ_API_KEY not set — add to .env to run live tests")
        return

    if test in ("all", "stt"):
        await test_stt()
    if test in ("all", "llm"):
        await test_llm()
    if test in ("all", "tts") and os.environ.get("CARTESIA_API_KEY"):
        await test_tts()
    elif test == "tts":
        print("[SKIP] CARTESIA_API_KEY not set")
    if test in ("all", "full"):
        await test_full_pipeline()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    asyncio.run(main(arg))
