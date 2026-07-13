# Echo — System Architecture

Echo runs the same **parallel dual-pipeline architecture** used by commercial
language-coaching products (Speak, TalkPal, ELSA): a fast conversational track
that always answers immediately, plus isolated analysis tracks that grade the
learner in the background and overlay feedback asynchronously.

```
                     ┌─► TRACK A1: Grammar analyzer (Groq Llama-8B, IELTS rubric,
                     │      95% confidence gate) ─► correction_card + errors table
                     │
                     ├─► TRACK A2: Audio delivery analyzer (Gemini Flash, hears the
                     │      ACTUAL audio + transcript) ─► per-word pronunciation
                     │      heatmap + live delivery tips + confidence/tone notes
[User audio ── WS] ──┤
                     ├─► TRACK A3: Deterministic metrics (Whisper timings →
                     │      WPM, filler counts — math, not model guesses)
                     │
                     └─► TRACK B: CONVERSATION (never blocked by A-tracks)
                            Groq Whisper STT ─► Llama-3.3-70B (streaming) ─►
                            sentence queue ─► TTS worker (Edge / Cartesia)
                            LLM generation overlaps TTS synthesis.

  All tracks converge ─► end-of-session scorecard (70B, JSON): overall score,
  CEFR estimate (auto-updates profile), STAR/content scores, strengths,
  improvements, model answers — grounded in A2/A3 measured data.
```

## Design rules

1. **Track B is sacred.** Analysis tracks are fire-and-forget `asyncio` tasks
   with their own timeouts; any of them failing, timing out, or being
   unconfigured (e.g. no `GEMINI_API_KEY`) never delays or breaks the spoken
   reply.
2. **Async UI overlay.** Cards/heatmaps arrive over the same WebSocket whenever
   they're ready — the app paints them onto the live transcript after the
   reply is already playing (`correction_card`, `pronunciation`, `ai_text`,
   `user_text` messages).
3. **Measured beats guessed.** WPM/fillers come from audio timings; the
   scorecard is instructed to use those numbers verbatim. The LLM only judges
   what genuinely requires judgment.
4. **Free-tier first.** Every engine has a $0 default (Groq free tier, Edge
   TTS, Gemini free tier, Supabase/Render free). Paid upgrades are env-var
   swaps, never code changes: `DEV_TTS=cartesia` (streaming ~100ms-class TTS),
   `GEMMA_AUDIO_MODEL`, future streaming STT.

## Client (Expo / Android)

- Recording: m4a/AAC 16kHz mono (universal since API 16) through
  `voice_communication` source → platform acoustic echo cancellation → the AI
  can listen while speaking (hands-free barge-in; talking over it interrupts).
- VAD: amplitude metering; during AI playback speech must be +10dB and ~256ms
  sustained to count (echo-residue guard).
- Playback: sequential segment queue — overlapping voices are structurally
  impossible.
- Live transcript UI: caption bubbles per turn; pronunciation heatmap colors
  flagged words when Track A2 lands.

## Honest deltas vs. big-budget stacks (Speak-class)

| Capability | Echo | Speak-class | Path |
|---|---|---|---|
| Turn latency | ~3–6s (batch STT + free TTS) | <1s (streaming STT + Cartesia/WebRTC) | Deepgram streaming + `DEV_TTS=cartesia` + LiveKit/Pipecat transport when budget allows |
| Phoneme scoring | Model-judged per-word verdicts (audio-native) | Wav2Vec2 + forced alignment + GOP (deterministic) | Separate GPU microservice later; won't fit 512MB free dyno |
| Orchestration | Custom FastAPI/WS (lean, coaching-aware) | LiveKit Agents / Pipecat | Adopt when moving to WebRTC; they don't provide the coaching layer |

## Scaling notes

- Stateless per-connection: any number of Render instances behind a LB works;
  session state lives in-process per WS + Supabase for persistence.
- Quota guards: 30 sessions/user/day, 30-min/100-turn session caps, 15MB
  audio buffer cap.
- Observability: every stage logs status+latency to `app_events`; the `/admin`
  dashboard graphs error rates and latency per stage.
