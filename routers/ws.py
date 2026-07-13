"""
WebSocket session handler — free-tier cascade (Groq Whisper + Groq Llama 3.3 70B
+ Edge TTS). Turn-based, latency-tolerant by design: this is interview / speech /
professional practice, where the value is the quality of the feedback, not
sub-second turn-taking.

Per turn:  user audio -> Whisper STT -> Llama streams the next interviewer/coach
turn -> sentences piped to Edge TTS -> PCM streamed to the client.

On session end: the full transcript + measured delivery metrics go to the
feedback engine, which returns a structured scorecard (scores, STAR analysis,
filler/pace stats, model-answer rewrites) — the part users pay for.

Everything runs on free tiers: no GPU, no OpenAI, no Cartesia.

Mobile wire protocol (unchanged):
  client -> server : audio_chunk {data}, turn_end, interrupt, session_end
  server -> client : session_ready, tts_chunk {data}, tts_done, tts_stop,
                     session_feedback {report}, session_summary, error
Audio: PCM float32 LE, 24 kHz mono, base64 on the wire.
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState

from db.supabase import get_supabase, verify_token
from services.stt import transcribe_audio_full
from services.llm import stream_conversation, run_correction_check
from services.tts import stream_tts
from services.feedback import compute_delivery_metrics, generate_session_feedback
from services.gemma_audio import analyze_delivery, format_delivery_notes
from services.events import log_event
from prompts.coach import build_coach_prompt
from prompts.interview import build_interview_prompt
from prompts.speech import build_speech_prompt
from prompts.professional import build_professional_prompt
from prompts.conversation import build_conversation_prompt

router = APIRouter()
logger = logging.getLogger("echo.ws")

SENTENCE_ENDS = {".", "!", "?"}
MIN_SENTENCE_LEN = 25

# ── Abuse / resource limits (protect free-tier quotas and server memory) ──────
MAX_AUDIO_BUFFER_BYTES = 15 * 1024 * 1024        # 15 MB per turn — beyond this the turn is dropped
MAX_SESSION_SECONDS = int(os.environ.get("MAX_SESSION_SECONDS", "1800"))   # 30 min auto-end
MAX_TURNS_PER_SESSION = int(os.environ.get("MAX_TURNS_PER_SESSION", "100"))
MAX_SESSIONS_PER_DAY = int(os.environ.get("MAX_SESSIONS_PER_DAY", "30"))   # per user


async def send_json(ws: WebSocket, data: dict):
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(data))


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("utf-8")


def build_system_prompt(mode: str, role: str, cefr: str, accent: str,
                        top_errors: Optional[list] = None) -> str:
    # Single-mode product: the app always sends "coach" (adaptive conversation
    # coaching), grounded in the coaching knowledge base + this learner's data.
    # Older modes remain callable for API compatibility.
    if mode == "interview":
        return build_interview_prompt(domain=role or "general")
    if mode == "speech":
        return build_speech_prompt()
    if mode == "professional":
        return build_professional_prompt(cefr, accent)
    if mode == "conversation":
        return build_conversation_prompt(cefr, accent)
    # 'role' doubles as the session focus for coach mode (general/interview/...).
    return build_coach_prompt(cefr, accent, focus=role or "general", top_errors=top_errors)


@router.websocket("/ws/{session_id}")
async def websocket_session(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
    mode: str = Query("coach"),
    accent: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    await websocket.accept()

    if not token:
        await send_json(websocket, {"type": "error", "message": "Missing token"})
        await websocket.close(code=4001, reason="Missing token")
        return

    user = verify_token(token)
    if not user:
        await send_json(websocket, {"type": "error", "message": "Invalid or expired token"})
        await websocket.close(code=4001, reason="Invalid token")
        return

    user_id = user["id"]
    supabase = get_supabase()

    # Load profile (best-effort).
    try:
        user_res = supabase.table("users").select("*").eq("id", user_id).single().execute()
        user_data = user_res.data or {}
    except Exception:
        user_data = {}
    cefr = user_data.get("cefr_level") or "B1"
    accent = accent or user_data.get("preferred_accent", "american")

    # Per-user daily session cap — protects the free-tier AI quotas from abuse.
    try:
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cnt = (
            supabase.table("sessions").select("id", count="exact")
            .eq("user_id", user_id).gte("started_at", day_start.isoformat())
            .execute().count or 0
        )
        if cnt >= MAX_SESSIONS_PER_DAY:
            await send_json(websocket, {
                "type": "error",
                "message": "Daily practice limit reached — come back tomorrow!",
            })
            await websocket.close(code=4008, reason="Daily limit")
            return
    except Exception:
        pass  # cap check must never block a legitimate session

    # Session row.
    session_db_id = str(uuid.uuid4())
    session_started_at = time.time()
    try:
        supabase.table("sessions").insert({"id": session_db_id, "user_id": user_id, "mode": mode}).execute()
    except Exception as e:
        logger.error("session insert failed: %s", e)
        await send_json(websocket, {"type": "error", "message": "Could not start the session. Please try again."})
        await websocket.close()
        return
    log_event("session_started", user_id=user_id, session_id=session_db_id, detail={"mode": mode})

    # Ground the coach in this learner's real recurring weak spots.
    top_errors: list[str] = []
    try:
        er = (
            supabase.table("errors")
            .select("error_type")
            .eq("user_id", user_id)
            .eq("resolved", False)
            .order("recurrence_count", desc=True)
            .limit(4)
            .execute()
        )
        top_errors = [row["error_type"] for row in (er.data or []) if row.get("error_type")]
    except Exception:
        pass

    system_prompt = build_system_prompt(mode, role, cefr, accent, top_errors)

    # In-memory session state (no Redis needed for a single connection).
    history: list[dict] = []
    audio_buffer = bytearray()
    speaking = {"on": False}
    gen_task: Optional[asyncio.Task] = None

    # Accumulating delivery metrics across the session.
    metrics_acc = {"words": 0, "seconds": 0.0, "fillers": {}}

    # Gemma 4 audio-native delivery analysis (sidecar — never blocks the reply).
    delivery_notes: list[dict] = []
    tip_state = {"turn": 0, "last_tip_turn": -2}
    # Proactive silence nudge bookkeeping.
    nudge_state = {"last_ai_done": 0.0, "nudged": False}

    NUDGES = {
        "coach": "No rush — take your time. What do you think?",
        "interview": "Take your time — walk me through it whenever you're ready.",
        "speech": "Whenever you're ready — take a breath and start speaking.",
        "professional": "No rush — gather your thoughts and go ahead.",
        "conversation": "Still there? I'd love to hear what you think.",
    }

    await send_json(websocket, {"type": "session_ready", "mode": mode, "session_id": session_db_id})

    # ── Stream one AI turn (LLM tokens -> sentence queue -> TTS worker) ───────
    # Producer/consumer overlap: the LLM keeps generating sentence N+1 while the
    # TTS worker is still synthesizing/streaming sentence N. This hides all TTS
    # latency except the very first sentence — the single biggest free-tier
    # latency win available (no engine change needed).
    async def speak_ai_turn(user_message: Optional[str]) -> None:
        speaking["on"] = True
        sentence_buf = ""
        full = ""
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def tts_worker() -> None:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    return
                if not speaking["on"]:
                    continue  # drain remaining items after interruption
                await send_json(websocket, {"type": "ai_text", "text": sentence})
                await _pipe_tts(sentence)

        worker = asyncio.create_task(tts_worker())
        try:
            async for token in stream_conversation(system_prompt, history, user_message):
                if not speaking["on"]:
                    break
                sentence_buf += token
                full += token
                if any(sentence_buf.rstrip().endswith(e) for e in SENTENCE_ENDS) and len(sentence_buf.strip()) >= MIN_SENTENCE_LEN:
                    chunk, sentence_buf = sentence_buf.strip(), ""
                    queue.put_nowait(chunk)
            if speaking["on"] and sentence_buf.strip():
                queue.put_nowait(sentence_buf.strip())
            queue.put_nowait(None)
            await worker
            if speaking["on"]:
                await send_json(websocket, {"type": "tts_done"})
        except Exception as e:
            log_event("llm", status="error", user_id=user_id, session_id=session_db_id, detail={"error": str(e)})
            logger.error("AI turn error: %s", e)
            await send_json(websocket, {"type": "error", "message": "The coach hit a snag — please continue."})
        finally:
            if not worker.done():
                queue.put_nowait(None)
                try:
                    await worker
                except Exception:
                    pass
            speaking["on"] = False
            nudge_state["last_ai_done"] = time.time()
            nudge_state["nudged"] = False
        if full.strip():
            history.append({"role": "assistant", "content": full.strip()})

    async def _pipe_tts(sentence: str) -> bool:
        async for audio_chunk in stream_tts(sentence, accent=accent):
            if not speaking["on"]:
                return False
            if audio_chunk:
                await send_json(websocket, {"type": "tts_chunk", "data": _b64(audio_chunk)})
        return True

    async def stop_speaking() -> None:
        nonlocal gen_task
        speaking["on"] = False
        if gen_task is not None and not gen_task.done():
            try:
                await gen_task
            except Exception:
                pass
        gen_task = None

    async def run_grammar_check(utterance: str) -> None:
        """Stream C: live grammar correction on the user's words. Fire-and-forget —
        the card slides in while the AI is already speaking its reply."""
        t0 = time.time()
        try:
            async with asyncio.timeout(8.0):
                result = await run_correction_check(utterance)
        except Exception:
            return
        if not result:
            return
        log_event("correction", user_id=user_id, session_id=session_db_id,
                  latency_ms=int((time.time() - t0) * 1000),
                  detail={"error_type": result.get("error_type")})
        await send_json(websocket, {
            "type": "correction_card",
            "error": {
                "error_type": result.get("error_type", "VOCAB"),
                "original_utterance": result.get("original", ""),
                "corrected_form": result.get("corrected", ""),
                "explanation": result.get("explanation", ""),
                "full_sentence": result.get("better_version", ""),
            },
        })
        # Persist for the progress/error-history screens (recurrence-aware).
        try:
            existing = (
                supabase.table("errors")
                .select("id, recurrence_count")
                .eq("user_id", user_id)
                .eq("original_utterance", result.get("original", ""))
                .eq("error_type", result.get("error_type", ""))
                .limit(1)
                .execute()
            )
            if existing.data:
                supabase.table("errors").update({
                    "recurrence_count": existing.data[0]["recurrence_count"] + 1,
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                }).eq("id", existing.data[0]["id"]).execute()
            else:
                supabase.table("errors").insert({
                    "user_id": user_id,
                    "session_id": session_db_id,
                    "error_type": result.get("error_type", "VOCAB"),
                    "original_utterance": result.get("original", ""),
                    "corrected_form": result.get("corrected", ""),
                    "explanation": result.get("explanation", ""),
                }).execute()
        except Exception:
            pass  # history persistence is best-effort

    async def run_delivery_analysis(audio_bytes: bytes, duration_sec: float, transcript: str = "") -> None:
        """Gemma 4 sidecar: hear the answer, store the judgment, maybe coach live."""
        t0 = time.time()
        note = await analyze_delivery(audio_bytes, duration_sec, transcript)
        if not note:
            return
        delivery_notes.append(note)
        log_event("gemma_audio", user_id=user_id, session_id=session_db_id,
                  latency_ms=int((time.time() - t0) * 1000),
                  detail={"confidence": note.get("confidence"), "hesitancy": note.get("hesitancy")})

        # Pronunciation heatmap: per-word verdicts judged from the ACTUAL audio.
        # The app colors these words in the user's caption bubble when this
        # lands (async overlay — arrives after the reply, never blocks it).
        words = note.get("word_scores")
        if isinstance(words, list) and words and transcript:
            clean = [
                {"word": w.get("word", ""), "issue": w.get("issue", "")}
                for w in words
                if isinstance(w, dict) and w.get("word")
            ][:4]
            if clean:
                await send_json(websocket, {
                    "type": "pronunciation",
                    "text": transcript,
                    "words": clean,
                })

        # Proactive live tip — throttled to at most one every 2 turns so the
        # coach feels attentive, not naggy. Rendered by the existing card UI.
        tip = (note.get("delivery_tip") or "").strip()
        if tip and tip_state["turn"] - tip_state["last_tip_turn"] >= 2:
            tip_state["last_tip_turn"] = tip_state["turn"]
            await send_json(websocket, {
                "type": "correction_card",
                "error": {
                    "error_type": "DELIVERY",
                    "original_utterance": note.get("tone", "your delivery"),
                    "corrected_form": tip,
                    "explanation": tip,
                },
            })

    async def run_silence_nudge() -> None:
        """Proactive coach: if the user stays silent well after the AI finishes,
        speak a short encouragement instead of dead air.

        The threshold is measured from when the last audio chunk was SENT — the
        client may still be *playing* that audio for another 10-15s, so this must
        be generous or the nudge overlaps the AI's own voice on the phone."""
        while True:
            await asyncio.sleep(2.0)
            last = nudge_state["last_ai_done"]
            if (
                last > 0
                and not nudge_state["nudged"]
                and not speaking["on"]
                and not audio_buffer
                and time.time() - last > 30.0
            ):
                nudge_state["nudged"] = True
                speaking["on"] = True
                try:
                    text = NUDGES.get(mode, NUDGES["conversation"])
                    await send_json(websocket, {"type": "ai_text", "text": text})
                    async for chunk in stream_tts(text, accent=accent):
                        if not speaking["on"]:
                            break
                        if chunk:
                            await send_json(websocket, {"type": "tts_chunk", "data": _b64(chunk)})
                    if speaking["on"]:
                        await send_json(websocket, {"type": "tts_done"})
                except Exception:
                    pass
                finally:
                    speaking["on"] = False
                    nudge_state["last_ai_done"] = time.time()  # don't re-nudge immediately

    async def run_feedback_and_save() -> None:
        metrics = compute_delivery_metrics(metrics_acc["words"], metrics_acc["seconds"], metrics_acc["fillers"])
        report = None
        t_fb = time.time()
        # Let any in-flight audio analyses land so the scorecard reflects the
        # last turn too. Gemini audio judgment runs ~15s, so give it room —
        # this wait happens behind the app's "Analyzing…" screen, not mid-chat.
        if analysis_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*analysis_tasks, return_exceptions=True), timeout=16.0)
            except asyncio.TimeoutError:
                pass
        try:
            report = await generate_session_feedback(mode, history, metrics, delivery_notes)
            log_event("feedback", status="ok" if report else "skipped",
                      user_id=user_id, session_id=session_db_id,
                      latency_ms=int((time.time() - t_fb) * 1000))
        except Exception as e:
            log_event("feedback", status="error", user_id=user_id, session_id=session_db_id, detail={"error": str(e)})
            report = None

        transcript_text = "\n".join(
            f"{'AI' if t['role'] == 'assistant' else 'User'}: {t['content']}" for t in history
        )
        duration_seconds = int(time.time() - session_started_at)
        update = {
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "transcript": transcript_text,
            "duration_seconds": duration_seconds,
            "wpm": metrics.get("pace_wpm"),
            "filler_count": metrics.get("filler_count"),
        }
        if report:
            update["overall_score"] = report.get("overall_score")
            update["feedback_json"] = report
        try:
            supabase.table("sessions").update(update).eq("id", session_db_id).execute()
        except Exception:
            # Migration-tolerant fallback: if the scorecard columns don't exist
            # yet (add_admin.sql not applied), still persist the core fields so
            # progress/streak/history keep working.
            try:
                supabase.table("sessions").update({
                    "ended_at": update["ended_at"],
                    "transcript": update["transcript"],
                    "duration_seconds": update["duration_seconds"],
                }).eq("id", session_db_id).execute()
            except Exception:
                logger.error("session update failed for %s", session_db_id)

        # Live level tracking: the scorecard estimates CEFR each session — keep
        # the user profile current so the home level card is real, not mock.
        cefr_est = (report or {}).get("cefr_estimate")
        if cefr_est in ("A1", "A2", "B1", "B2", "C1", "C2"):
            try:
                supabase.table("users").update({
                    "cefr_level": cefr_est,
                    "last_active": datetime.now(timezone.utc).isoformat(),
                }).eq("id", user_id).execute()
            except Exception:
                pass

        log_event("session_completed", user_id=user_id, session_id=session_db_id,
                  latency_ms=duration_seconds * 1000,
                  detail={"mode": mode, "turns": len(history), "scored": bool(report),
                          "overall_score": report.get("overall_score") if report else None})

        if report:
            await send_json(websocket, {"type": "session_feedback", "report": report, "metrics": metrics})
        await send_json(websocket, {"type": "session_summary", "session_id": session_db_id})

    nudge_task = asyncio.create_task(run_silence_nudge())
    analysis_tasks: list[asyncio.Task] = []

    try:
        # AI opens the session (intro + first question/prompt).
        gen_task = asyncio.create_task(speak_ai_turn(None))

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                # Cap per-turn audio to protect memory + the free STT quota.
                if len(audio_buffer) < MAX_AUDIO_BUFFER_BYTES:
                    try:
                        audio_buffer.extend(base64.b64decode(msg.get("data", "")))
                    except Exception:
                        pass  # ignore malformed chunks

            elif msg_type == "turn_end":
                if not audio_buffer:
                    continue

                # Hard session limits: duration + turn count.
                if time.time() - session_started_at > MAX_SESSION_SECONDS or tip_state["turn"] >= MAX_TURNS_PER_SESSION:
                    audio_buffer.clear()
                    await send_json(websocket, {
                        "type": "error",
                        "message": "Session limit reached — ending with your scorecard now.",
                    })
                    await stop_speaking()
                    await run_feedback_and_save()
                    break

                audio_bytes = bytes(audio_buffer)
                audio_buffer.clear()

                await stop_speaking()

                t_stt = time.time()
                try:
                    stt = await transcribe_audio_full(audio_bytes)
                    log_event("stt", user_id=user_id, session_id=session_db_id,
                              latency_ms=int((time.time() - t_stt) * 1000))
                except Exception as e:
                    log_event("stt", status="error", user_id=user_id, session_id=session_db_id, detail={"error": str(e)})
                    await send_json(websocket, {"type": "error", "message": "Didn't catch that — please try speaking again."})
                    continue
                transcript = stt.text.strip()
                if not transcript:
                    continue

                # Show the user their own recognized words in the live transcript.
                await send_json(websocket, {"type": "user_text", "text": transcript})

                # Accumulate delivery metrics.
                metrics_acc["words"] += len(transcript.split())
                metrics_acc["seconds"] += stt.duration_sec
                for w, n in stt.filler_hits.items():
                    metrics_acc["fillers"][w] = metrics_acc["fillers"].get(w, 0) + n

                history.append({"role": "user", "content": transcript})
                tip_state["turn"] += 1
                nudge_state["nudged"] = True  # user spoke — cancel pending nudge this turn
                # Sidecars run in parallel — neither blocks the spoken reply:
                # grammar check (correction card) + Gemma 4 delivery analysis.
                analysis_tasks.append(asyncio.create_task(run_grammar_check(transcript)))
                analysis_tasks.append(asyncio.create_task(
                    run_delivery_analysis(audio_bytes, stt.duration_sec, transcript)))
                gen_task = asyncio.create_task(speak_ai_turn(None))

            elif msg_type == "interrupt":
                speaking["on"] = False
                await send_json(websocket, {"type": "tts_stop"})

            elif msg_type == "session_end":
                await stop_speaking()
                await run_feedback_and_save()
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("session loop error: %s", e)
        try:
            await send_json(websocket, {"type": "error", "message": "Connection issue — please rejoin the session."})
        except Exception:
            pass
    finally:
        speaking["on"] = False
        nudge_task.cancel()
        for t in analysis_tasks:
            if not t.done():
                t.cancel()
        if gen_task is not None and not gen_task.done():
            gen_task.cancel()
            try:
                await gen_task
            except Exception:
                pass
        try:
            supabase.table("sessions").update({
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", session_db_id).is_("ended_at", "null").execute()
        except Exception:
            pass
