import asyncio
import base64
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState

from db.supabase import get_supabase, verify_token
from services.redis_state import get_session_state, set_session_state, delete_session_state
from services.stt import transcribe_audio, transcribe_audio_full
from services.llm import stream_conversation, stream_diagnostic_score, stream_correction_check, generate_session_report
from services.tts import stream_tts
from services.pronunciation import score_from_whisper
from services.memory import load_user_memory, build_memory_context, update_user_memory
from services.curriculum import build_session_plan, build_accent_session_context
from services.entitlements import can_start_session
from services.gamification import award_session
from prompts.diagnostic import DIAGNOSTIC_SYSTEM, DIAGNOSTIC_CONVERSATION_SYSTEM
from prompts.training import build_training_prompt
from prompts.interview import build_interview_prompt
from prompts.conversation import build_conversation_prompt
from prompts.professional import build_professional_prompt
from prompts.accent import build_accent_prompt
from models.session import SessionState

router = APIRouter()


async def send_json(ws: WebSocket, data: dict):
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(data))


async def run_silence_nudge(ws: WebSocket, session_id: str):
    """Background task: send a nudge if user has been silent > 3s and AI is not speaking."""
    while True:
        await asyncio.sleep(0.5)
        try:
            state = await get_session_state(session_id)
            if state is None:
                break
            if state.get("tts_is_speaking"):
                continue
            silence_start = state.get("silence_timer_start")
            if silence_start and (time.time() - silence_start) > 3.0:
                state["silence_timer_start"] = None
                await set_session_state(session_id, state)
                await send_json(ws, {"type": "nudge"})
        except Exception:
            break


async def _pipe_sentence_to_tts(
    ws: WebSocket,
    session_id: str,
    sentence: str,
    accent: str,
) -> bool:
    """
    Send one sentence chunk to Cartesia and stream audio chunks to client.
    Returns False if interrupted (tts_is_speaking flipped to False by interrupt message).
    Never buffers full TTS response — streams each audio chunk as it arrives.
    """
    async for audio_chunk in stream_tts(sentence, accent=accent):
        if not audio_chunk:
            continue
        current_state = await get_session_state(session_id)
        if current_state is None or not current_state.get("tts_is_speaking"):
            return False  # interrupted
        await send_json(ws, {
            "type": "tts_chunk",
            "data": base64.b64encode(audio_chunk).decode("utf-8"),
        })
    return True


async def handle_stream_b(
    ws: WebSocket,
    session_id: str,
    transcript: str,
    state: dict,
):
    """
    Stream B: LLM tokens → sentence chunks → Cartesia TTS → client audio.

    Streaming pipeline (no buffering):
      LLM token arrives → append to sentence buffer
      Sentence boundary hit → fire that sentence to Cartesia immediately
      Cartesia audio chunk arrives → forward to client immediately

    This achieves < 750ms total latency:
      STT(200) + LLM first token(200) + first sentence assembled(~100) + Cartesia first chunk(300)
    """
    supabase = get_supabase()
    mode = state.get("mode", "training")
    turn_count = state.get("diagnostic_turn_count", 0)
    accent = state.get("preferred_accent", "american")

    try:
        user_res = supabase.table("users").select("cefr_level, preferred_accent").eq("id", state["user_id"]).single().execute()
        user_data = user_res.data or {}
    except Exception:
        user_data = {}

    cefr = state.get("cefr_level") or user_data.get("cefr_level") or "B1"
    # Prefer the accent chosen for THIS session (pre-session screen / WS param);
    # fall back to the user's saved default. Keeps the greeting and the rest of the
    # conversation in the same voice.
    accent = state.get("preferred_accent") or user_data.get("preferred_accent", "american")

    # Build memory context and curriculum context for this turn
    user_memory = state.get("user_memory", {})
    user_name = (user_data.get("full_name") or "").strip().split()[0] if user_data.get("full_name") else ""
    memory_ctx = build_memory_context(user_memory, user_name)

    top_errors = state.get("top_recurring_errors", [])
    current_turn = state.get("diagnostic_turn_count", 0)
    phoneme_unit = state.get("phoneme_unit", 0)

    # Curriculum context — tells the LLM what to focus on this specific turn
    curriculum_ctx = build_session_plan(
        mode=mode,
        cefr=cefr,
        accent=accent,
        memory=user_memory,
        turn_count=current_turn,
        top_errors=top_errors,
        phoneme_unit=phoneme_unit if mode == "accent" else None,
    )

    if mode == "diagnostic":
        # First session: keep Echo on the goal-discovery script — no generic
        # curriculum framing. (Memory is empty on a first session anyway.)
        system_prompt = DIAGNOSTIC_CONVERSATION_SYSTEM
    elif mode == "interview":
        system_prompt = curriculum_ctx + memory_ctx + build_interview_prompt()
    elif mode == "conversation":
        system_prompt = curriculum_ctx + memory_ctx + build_conversation_prompt(cefr, accent)
    elif mode == "professional":
        system_prompt = curriculum_ctx + memory_ctx + build_professional_prompt(cefr, accent)
    elif mode == "accent":
        exam_target = (user_data.get("exam_target") or "general").lower()
        accent_context_map = {
            "ielts": "ielts", "toefl": "toefl",
            "job interview": "interview", "interview": "interview",
        }
        ctx = accent_context_map.get(exam_target, "daily")
        phoneme_ctx = build_accent_session_context(accent, phoneme_unit, current_turn)
        system_prompt = phoneme_ctx + memory_ctx + build_accent_prompt(
            accent, cefr, context=ctx, curriculum_context=""
        )
    else:  # training (default)
        topic = state.get("current_topic", "general")
        system_prompt = curriculum_ctx + memory_ctx + build_training_prompt(cefr, top_errors, topic, accent)

    # If this session is a step in the learner's plan, steer it toward that focus.
    plan_focus = state.get("plan_focus", "")
    if plan_focus and mode != "diagnostic":
        system_prompt = plan_focus + system_prompt

    # Sentence boundary characters — flush to TTS when hit
    SENTENCE_ENDS = {'.', '!', '?'}
    MIN_SENTENCE_LEN = 30  # avoid firing TTS on fragments like "Right." or "I see."

    sentence_buf = ""
    full_response = ""
    interrupted = False
    assessment_complete = False

    try:
        async for token in stream_conversation(system_prompt, transcript, state):
            sentence_buf += token
            full_response += token

            # Check if diagnostic signals assessment complete
            if mode == "diagnostic" and "[ASSESSMENT_COMPLETE]" in full_response:
                full_response = full_response.replace("[ASSESSMENT_COMPLETE]", "").strip()
                sentence_buf = sentence_buf.replace("[ASSESSMENT_COMPLETE]", "").strip()
                # Speak whatever closing words came before the sentinel, then score.
                assessment_complete = True
                if sentence_buf.strip():
                    await _pipe_sentence_to_tts(ws, session_id, sentence_buf.strip(), accent)
                    sentence_buf = ""
                break

            # Flush to TTS on natural sentence boundary
            if (
                any(sentence_buf.rstrip().endswith(e) for e in SENTENCE_ENDS)
                and len(sentence_buf.strip()) >= MIN_SENTENCE_LEN
            ):
                chunk = sentence_buf.strip()
                sentence_buf = ""
                ok = await _pipe_sentence_to_tts(ws, session_id, chunk, accent)
                if not ok:
                    interrupted = True
                    break

        # Flush any trailing text (e.g. response ended without punctuation)
        if not interrupted and sentence_buf.strip():
            await _pipe_sentence_to_tts(ws, session_id, sentence_buf.strip(), accent)

        if not interrupted:
            await send_json(ws, {"type": "tts_done"})

        # Persist AI turn to conversation history
        if full_response:
            current_state = await get_session_state(session_id)
            if current_state:
                current_state["conversation_history"].append({"role": "assistant", "content": full_response})
                current_state["session_transcript"] += f"\nAssistant: {full_response}"
                await set_session_state(session_id, current_state)

    except Exception as e:
        await send_json(ws, {"type": "error", "message": f"Stream B error: {str(e)}"})
    finally:
        refreshed = await get_session_state(session_id)
        if refreshed:
            refreshed["tts_is_speaking"] = False
            refreshed["silence_timer_start"] = time.time()
            await set_session_state(session_id, refreshed)

    # Diagnostic scoring: trigger after turn 6 (the goal-discovery + assessment flow).
    # Two conditions — either the [ASSESSMENT_COMPLETE] sentinel OR a turn-count
    # fallback (guards against the LLM not emitting the sentinel).
    current_state = await get_session_state(session_id)
    diagnostic_done = (current_state or state).get("is_diagnostic_complete", False)
    if mode == "diagnostic" and (assessment_complete or turn_count >= 6) and not diagnostic_done:
        await trigger_diagnostic_score(ws, session_id, current_state or state)


async def handle_stream_c(
    ws: WebSocket,
    session_id: str,
    transcript: str,
    state: dict,
):
    """Stream C: Fire-and-forget error check (500ms budget)."""
    try:
        async with asyncio.timeout(0.5):
            result = await stream_correction_check(transcript)
            if result and result.get("error"):
                await send_json(ws, {
                    "type": "correction_card",
                    "error": result,
                })
                # Persist error to Supabase
                supabase = get_supabase()
                existing = (
                    supabase.table("errors")
                    .select("id, recurrence_count")
                    .eq("user_id", state["user_id"])
                    .eq("original_utterance", result.get("original", ""))
                    .eq("error_type", result.get("error_type", ""))
                    .limit(1)
                    .execute()
                )
                if existing.data:
                    err_id = existing.data[0]["id"]
                    new_count = existing.data[0]["recurrence_count"] + 1
                    supabase.table("errors").update({
                        "recurrence_count": new_count,
                        "last_seen": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", err_id).execute()
                else:
                    supabase.table("errors").insert({
                        "user_id": state["user_id"],
                        "session_id": state.get("session_db_id"),
                        "error_type": result.get("error_type", "UNKNOWN"),
                        "original_utterance": result.get("original", transcript),
                        "corrected_form": result.get("corrected", ""),
                        "explanation": result.get("explanation", ""),
                    }).execute()

                # Update session error count
                state["error_count_this_session"] = state.get("error_count_this_session", 0) + 1
                await set_session_state(session_id, state)
    except asyncio.TimeoutError:
        pass  # Silently drop — never block Stream B
    except Exception:
        pass  # Never let Stream C crash the session


def _get_plan_focus(supabase, user_id: str, mode: str) -> str:
    """
    If the user has an active roadmap and their next pending session matches the
    mode they just started, return a context block that steers this session toward
    that plan session's focus/skills. Returns "" otherwise. Best-effort, never raises.
    """
    if mode == "diagnostic":
        return ""
    try:
        res = (
            supabase.table("learning_plans")
            .select("plan_json, generated_at")
            .eq("user_id", user_id)
            .order("generated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return ""
        roadmap = (res.data[0].get("plan_json") or {}).get("roadmap") or {}
        sessions = roadmap.get("sessions") or []
        done = int(roadmap.get("completed_count", 0))
        if not (0 <= done < len(sessions)):
            return ""
        nxt = sessions[done]
        if nxt.get("mode") != mode:
            return ""  # they started a different mode than the plan suggests — stay generic
        title = nxt.get("title", "")
        focus = nxt.get("focus", "")
        desc = nxt.get("description", "")
        skills = ", ".join(nxt.get("skills", []) or [])
        goal = roadmap.get("goal", "")
        lines = [
            "━━━ TODAY'S PLAN SESSION (internal — do not read aloud) ━━━",
            f"This is session {nxt.get('n', done + 1)} of the learner's personalised plan toward: {goal}.",
            f"Session title: {title}",
            f"Primary focus THIS session: {focus}",
        ]
        if skills:
            lines.append(f"Target skills to elicit and reinforce: {skills}")
        if desc:
            lines.append(f"Plan note: {desc}")
        lines.append(
            "Steer the whole conversation so the learner naturally produces and practises the focus above. "
            "Keep it conversational — never announce the lesson, never drill mechanically."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines) + "\n\n"
    except Exception:
        return ""


def _advance_roadmap(supabase, user_id: str) -> None:
    """
    Mark the next pending roadmap session as completed and bump completed_count
    inside the user's latest learning_plans.plan_json.
    Best-effort — never raises. Called when a non-diagnostic session ends.
    """
    try:
        res = (
            supabase.table("learning_plans")
            .select("id, plan_json")
            .eq("user_id", user_id)
            .order("generated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return
        row_id = res.data[0]["id"]
        plan = res.data[0].get("plan_json") or {}
        roadmap = plan.get("roadmap") or {}
        sessions = roadmap.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            return
        done = int(roadmap.get("completed_count", 0))
        if done >= len(sessions):
            return
        sessions[done]["status"] = "done"
        roadmap["completed_count"] = done + 1
        plan["roadmap"] = roadmap
        supabase.table("learning_plans").update({"plan_json": plan}).eq("id", row_id).execute()
    except Exception:
        pass


async def trigger_diagnostic_score(ws: WebSocket, session_id: str, state: dict):
    """
    Call Llama 3.3 70B for full IELTS-rubric-grounded scoring after 5 turns.
    Merges Whisper pronunciation proxy scores into the result if Azure is unavailable.
    """
    try:
        supabase = get_supabase()

        # Merge Whisper pronunciation proxy into transcript context
        pron_scores = state.get("pronunciation_scores", [])
        transcript = state.get("session_transcript", "")

        if pron_scores:
            avg_pron = sum(pron_scores) / len(pron_scores)
            pron_ielts = round(avg_pron / 100 * 9 * 2) / 2  # map to 0.5 steps
            transcript += (
                f"\n\n[SYSTEM NOTE FOR SCORER: Whisper pronunciation confidence proxy = "
                f"{avg_pron:.1f}/100 (≈ IELTS {pron_ielts}). "
                f"Use this as a data point for your pronunciation band, not the sole determinant.]"
            )

        result = await stream_diagnostic_score(transcript)
        if result:
            state["is_diagnostic_complete"] = True
            state["cefr_level"] = result.get("cefr_level")
            await set_session_state(session_id, state)

            roadmap = result.get("roadmap") or {}
            goal = (roadmap.get("goal") or result.get("goal") or "").strip()

            # Stamp the learner's preferred accent onto every plan session so
            # accent drills (and TTS voice) match what they signed up for.
            pref_accent = state.get("preferred_accent", "american")
            for _sess in roadmap.get("sessions", []):
                if isinstance(_sess, dict):
                    _sess["accent"] = pref_accent

            # Update user profile — level, IELTS estimate, and discovered goal.
            # (The full roadmap lives in learning_plans.plan_json, written below.)
            user_update = {
                "cefr_level": result.get("cefr_level"),
                "ielts_score": result.get("ielts_equivalent"),
                "last_active": datetime.now(timezone.utc).isoformat(),
            }
            if goal:
                user_update["goal"] = goal
            supabase.table("users").update(user_update).eq("id", state["user_id"]).execute()

            # Update session row with all subscores
            supabase.table("sessions").update({
                "cefr_score": result.get("cefr_level"),
                "fluency_score": result.get("fluency"),
                "lexical_score": result.get("lexical"),
                "grammar_score": result.get("grammar"),
                "pronunciation_score": result.get("pronunciation"),
            }).eq("id", state.get("session_db_id")).execute()

            # Save personalized learning plan (full result incl. roadmap)
            supabase.table("learning_plans").insert({
                "user_id": state["user_id"],
                "plan_json": result,
            }).execute()

            await send_json(ws, {"type": "diagnostic_complete", "result": result})
    except Exception as e:
        await send_json(ws, {"type": "error", "message": f"Diagnostic scoring error: {str(e)}"})


async def _run_post_session_report(
    session_db_id: str,
    user_id: str,
    transcript: str,
    cefr: str,
    audio_segments: list[dict],
) -> None:
    """
    Post-session slow track — runs after the WebSocket closes.
    1. Wav2Vec2 phoneme analysis on buffered audio segments
    2. Gemini Flash report generation (grammar + vocabulary + accent coaching)
    3. Save report JSON to Supabase sessions table
    Never blocks live conversation — runs fully async in background.
    """
    try:
        # Step 1: Wav2Vec2 phoneme analysis (CPU-bound, run in thread pool)
        phoneme_errors: list[dict] = []
        if audio_segments:
            loop = asyncio.get_event_loop()
            from services.phoneme import analyze_pronunciation_batch
            phoneme_errors = await loop.run_in_executor(
                None,
                analyze_pronunciation_batch,
                audio_segments,
                24000,  # original sample rate
            )

        # Step 2: LLM report generation
        report = await generate_session_report(
            transcript=transcript,
            phoneme_errors=phoneme_errors,
            cefr_level=cefr,
        )

        if report:
            # Step 3a: Try the dedicated column (if the DB has it).
            try:
                get_supabase().table("sessions").update({
                    "report_json": report,
                }).eq("id", session_db_id).execute()
            except Exception:
                pass
            # Step 3b: Always cache it in users.memory so the session-summary
            # endpoint can serve it even without the report_json column.
            try:
                from services.user_state import update_memory

                def _store(mem: dict):
                    reports = mem.get("reports")
                    if not isinstance(reports, dict):
                        reports = {}
                    reports[session_db_id] = report
                    # keep only the most recent 8 reports
                    if len(reports) > 8:
                        for k in list(reports.keys())[:-8]:
                            reports.pop(k, None)
                    mem["reports"] = reports

                update_memory(user_id, _store)
            except Exception:
                pass

    except Exception:
        pass  # Report generation is best-effort — never crash post-session


@router.websocket("/ws/{session_id}")
async def websocket_session(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
    mode: str = Query("training"),
    accent: Optional[str] = Query(None),
):
    # Must accept before closing — closing an unaccepted WS returns 403 to client.
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
    supabase = get_supabase()
    user_id = user["id"]

    # Step 1b: Entitlement gate — free tier limits (premium modes + daily cap).
    gate = can_start_session(user_id, mode)
    if not gate["allowed"]:
        await send_json(websocket, {
            "type": "upgrade_required",
            "reason": gate["reason"],          # "premium_mode" | "daily_limit"
            "mode": mode,
        })
        await websocket.close(code=4003, reason="Upgrade required")
        return

    # Step 2: Create session row in Supabase
    session_db_id = str(uuid.uuid4())
    try:
        supabase.table("sessions").insert({
            "id": session_db_id,
            "user_id": user_id,
            "mode": mode,
        }).execute()
    except Exception as e:
        await send_json(websocket, {"type": "error", "message": f"Session creation failed: {str(e)}"})
        await websocket.close()
        return

    # Steps 3-6: Load context + initialise Redis session state.
    # Wrapped so a backend dependency failure (e.g. REDIS_URL not set on the host,
    # Supabase outage) surfaces as a clear client error + server log instead of an
    # abnormal 1006 close that makes the client reconnect-loop forever.
    try:
        # Step 3: Load top recurring errors
        errors_res = (
            supabase.table("errors")
            .select("error_type, original_utterance, recurrence_count")
            .eq("user_id", user_id)
            .eq("resolved", False)
            .order("recurrence_count", desc=True)
            .limit(3)
            .execute()
        )
        top_errors = errors_res.data or []

        # Step 4: Load user profile
        user_res = supabase.table("users").select("*").eq("id", user_id).single().execute()
        user_data = user_res.data or {}

        # Step 5: Load user memory (persistent learning context across sessions)
        user_memory = load_user_memory(user_id)

        # Step 5b: If this session is the next step in the learner's plan, load its
        # focus so the whole conversation steers toward that plan session's topic.
        plan_focus = _get_plan_focus(supabase, user_id, mode)

        # Step 6: Initialize Redis session state
        state = {
            "user_id": user_id,
            "mode": mode,
            "is_diagnostic_complete": False,
            "diagnostic_turn_count": 0,
            "phoneme_unit": 0,               # accent mode: current phoneme curriculum unit
            "current_topic": user_memory.get("preferred_topics", ["career"])[0] if user_memory.get("preferred_topics") else "career",
            "silence_timer_start": None,
            "error_count_this_session": 0,
            "cefr_level": user_data.get("cefr_level"),
            "top_recurring_errors": top_errors,
            "tts_is_speaking": False,
            "session_db_id": session_db_id,
            "preferred_accent": accent or user_data.get("preferred_accent", "american"),
            "conversation_history": [],
            "session_transcript": "",
            "session_started_at": time.time(),
            "user_memory": user_memory,
            "plan_focus": plan_focus,        # plan session context, injected into the prompt each turn
            # Audio buffer for post-session Wav2Vec2 analysis (stores {audio_bytes, transcript, ts})
            # Not stored in Redis (too large) — kept in-memory for this connection lifetime
        }
        await set_session_state(session_id, state)
    except Exception as e:
        # Log the full traceback to the server (visible in Railway logs) and tell
        # the client exactly what failed so it can stop and show the error.
        print("[ws] FATAL session setup error:", repr(e))
        traceback.print_exc()
        await send_json(websocket, {
            "type": "fatal_error",
            "message": f"Session setup failed: {type(e).__name__}: {str(e)[:200]}",
        })
        await websocket.close(code=1011, reason="Session setup failed")
        return

    # In-memory audio segments buffer for post-session phoneme analysis
    # Each entry: {"audio_bytes": bytes, "expected_text": str, "timestamp": float}
    audio_segments_buffer: list[dict] = []

    # Step 7: Send session_ready + trigger AI greeting
    await send_json(websocket, {
        "type": "session_ready",
        "cefr": user_data.get("cefr_level"),
        "mode": mode,
        "session_id": session_db_id,
    })

    # Start silence nudge background task
    nudge_task = asyncio.create_task(run_silence_nudge(websocket, session_id))

    # Build personalised greeting using the user's name if available
    first_name = ""
    full_name = user_data.get("full_name", "")
    if full_name:
        first_name = full_name.strip().split()[0]

    name_part = f", {first_name}" if first_name else ""

    greetings = {
        "diagnostic":   f"Hi{name_part}, welcome — I'm Echo, your English coach. This first chat is just so I can get to know you and build you a plan. No test, no right answers — talk to me naturally. So tell me: what's bringing you to English right now, and what are you hoping to get out of it?",
        "training":     f"Hey{name_part}! Good to have you. Let's get straight into it — tell me, what's been keeping you busy lately?",
        "interview":    f"Thanks for coming in{name_part}. Let's jump straight to it — walk me through your background. Who are you and what do you do?",
        "conversation": f"Hey{name_part}! Great to chat. So — what's going on with you lately? Anything interesting happen this week?",
        "professional": f"Right{name_part}, let's get started. Give me a quick summary of what you're currently working on.",
        "accent":       f"Welcome{name_part}. We're going to work on your pronunciation through shadowing. I'll say a phrase, you repeat it back. Ready? Let's start simple: \"I need to think about it.\" Your turn — repeat that.",
    }
    greeting_text = greetings.get(mode, greetings["training"])

    state["tts_is_speaking"] = True
    state["conversation_history"].append({"role": "assistant", "content": greeting_text})
    await set_session_state(session_id, state)

    # Greeting audio. If TTS fails (network to the TTS provider, etc.) we must NOT
    # kill the session — the user can still talk and the AI replies per-turn. Send
    # the greeting text so the UI can show it, then always emit tts_done.
    await send_json(websocket, {"type": "assistant_text", "text": greeting_text})
    try:
        async for audio_chunk in stream_tts(greeting_text, accent=state["preferred_accent"]):
            await send_json(websocket, {
                "type": "tts_chunk",
                "data": base64.b64encode(audio_chunk).decode("utf-8"),
            })
    except Exception as e:
        print("[ws] greeting TTS failed (continuing):", repr(e))
        traceback.print_exc()

    await send_json(websocket, {"type": "tts_done"})
    state["tts_is_speaking"] = False
    state["silence_timer_start"] = time.time()
    await set_session_state(session_id, state)

    # ── Main message loop ────────────────────────────────────────────────────
    audio_buffer = bytearray()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                chunk_data = msg.get("data", "")
                audio_buffer.extend(base64.b64decode(chunk_data))

            elif msg_type == "turn_end":
                if not audio_buffer:
                    continue

                audio_bytes = bytes(audio_buffer)
                audio_buffer.clear()

                # Transcribe — use full result in diagnostic mode for pronunciation hints
                try:
                    current_mode = (await get_session_state(session_id) or {}).get("mode", mode)
                    if current_mode == "diagnostic":
                        stt_result = await transcribe_audio_full(audio_bytes, language="en", include_pronunciation_hints=True)
                        transcript = stt_result.text
                    else:
                        transcript = await transcribe_audio(audio_bytes)
                        stt_result = None
                except Exception as e:
                    await send_json(websocket, {"type": "error", "message": f"STT failed: {str(e)}"})
                    continue

                if not transcript.strip():
                    continue

                # Buffer audio for post-session phoneme analysis (accent + diagnostic modes)
                if mode in ("accent", "diagnostic") and len(audio_segments_buffer) < 20:
                    audio_segments_buffer.append({
                        "audio_bytes": audio_bytes,
                        "expected_text": transcript,
                        "timestamp": time.time(),
                    })

                state = await get_session_state(session_id)
                if state is None:
                    break

                # Update conversation history + transcript (cap at 20 turns to avoid Redis overflow)
                state["conversation_history"].append({"role": "user", "content": transcript})
                if len(state["conversation_history"]) > 60:
                    # Keep most recent 60 turns (30 exchanges) — older context pruned to stay within Redis limits
                    state["conversation_history"] = state["conversation_history"][-60:]
                state["session_transcript"] += f"\nUser: {transcript}"
                new_turn_count = state.get("diagnostic_turn_count", 0) + 1
                state["diagnostic_turn_count"] = new_turn_count
                state["tts_is_speaking"] = True
                state["silence_timer_start"] = None

                # Store pronunciation proxy data for diagnostic scoring
                if stt_result and stt_result.pronunciation_score is not None:
                    pron_scores = state.get("pronunciation_scores", [])
                    pron_scores.append(stt_result.pronunciation_score)
                    state["pronunciation_scores"] = pron_scores

                await set_session_state(session_id, state)

                # Fire Stream B and Stream C in parallel
                await asyncio.gather(
                    handle_stream_b(websocket, session_id, transcript, state),
                    handle_stream_c(websocket, session_id, transcript, state),
                )

            elif msg_type == "interrupt":
                state = await get_session_state(session_id)
                if state:
                    state["tts_is_speaking"] = False
                    await set_session_state(session_id, state)
                await send_json(websocket, {"type": "tts_stop"})

            elif msg_type == "session_end":
                state = await get_session_state(session_id)
                if state:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    final_transcript = state.get("session_transcript", "")
                    supabase.table("sessions").update({
                        "ended_at": now_iso,
                        "transcript": final_transcript,
                    }).eq("id", session_db_id).execute()

                    if state.get("is_diagnostic_complete"):
                        await send_json(websocket, {"type": "session_summary", "session_id": session_db_id})

                    # Advance the learning roadmap when a real (non-diagnostic) plan
                    # session is completed, so the home screen surfaces the next one.
                    if mode != "diagnostic":
                        _advance_roadmap(supabase, user_id)

                    session_minutes = int((time.time() - state.get("session_started_at", time.time())) / 60)

                    # Award XP / update streak + level (synchronous, quick) and tell
                    # the client so it can celebrate the gain.
                    try:
                        xp_result = await asyncio.get_event_loop().run_in_executor(
                            None, award_session, user_id, mode, max(1, session_minutes),
                            state.get("cefr_level"), session_db_id,
                        )
                        await send_json(websocket, {"type": "xp_awarded", "result": xp_result})
                    except Exception:
                        pass

                    # Fire post-session slow-track pipeline (non-blocking background tasks)
                    asyncio.create_task(update_user_memory(
                        user_id=user_id,
                        existing_memory=state.get("user_memory", {}),
                        session_transcript=final_transcript,
                        session_mode=mode,
                        session_minutes=max(1, session_minutes),
                        top_errors=state.get("top_recurring_errors", []),
                    ))

                    # Post-session report: Wav2Vec2 phoneme analysis + LLM report generation
                    asyncio.create_task(_run_post_session_report(
                        session_db_id=session_db_id,
                        user_id=user_id,
                        transcript=final_transcript,
                        cefr=state.get("cefr_level", "B1"),
                        audio_segments=list(audio_segments_buffer),  # copy before clearing
                    ))

                    await delete_session_state(session_id)

                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send_json(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        nudge_task.cancel()
        # Clean up: mark session ended if still open (e.g. network drop)
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            supabase.table("sessions").update({"ended_at": now_iso}).eq("id", session_db_id).is_("ended_at", "null").execute()
        except Exception:
            pass
        await delete_session_state(session_id)
