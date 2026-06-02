"""
Conversation simulator — drives Echo through a full multi-turn session using the
REAL prompt assembly + LLM path (no voice, no DB). Lets us read exactly how Echo
leads, reacts, and (for diagnostic) builds the roadmap, so we can tune prompts.

Usage:
  python -m eval.sim_conversation diagnostic
  python -m eval.sim_conversation training
  python -m eval.sim_conversation interview
"""
import asyncio
import sys
import warnings

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

from services.llm import stream_conversation, stream_diagnostic_score
from services.curriculum import build_session_plan, build_accent_session_context
from prompts.diagnostic import DIAGNOSTIC_CONVERSATION_SYSTEM
from prompts.training import build_training_prompt
from prompts.interview import build_interview_prompt
from prompts.conversation import build_conversation_prompt
from prompts.professional import build_professional_prompt
from prompts.accent import build_accent_prompt


# ── A realistic B1 learner with characteristic errors Echo should detect ──────
LEARNER_SCRIPTS = {
    "diagnostic": [
        "Hi, em, I want to improve my English because I am applying for a master degree in UK next year and I need IELTS band seven.",
        "It is important because the university require band seven, and honestly I am bit nervous about the speaking part, I never speak English with native people.",
        "Okay so, the most challenging thing was last year, we have a big project for a important client and the deadline was very tight. I am working as a software engineer since three years, but this project was different because I am leading it alone. The database was very slow and the client was angry, so every day I am fixing bugs and also explaining to the client in English on call, which made me very stressed. At the end we deliver it but two weeks late, and I learn that I should communicate the problems earlier instead of hiding them.",
        "I think remote work is better because you save time from commute, but some people they prefer office because is more easy to communicate. For me, I am more productive in home.",
        "If I could change one thing in my job, I will reduce the meetings. We have too much meeting every day and it waste a lot of time that I can use for coding.",
        "Yes exactly, thank you. I hope I can reach my goal soon.",
    ],
    "training": [
        "Hello, last weekend I go to the mountains with my friends.",
        "We are driving for three hours and then we did a hiking. The view was very beautiful.",
        "I am living in this city since 2019, but I never visit those mountains before.",
        "My friend he take many photos, and we eat lunch near a small lake.",
        "Next time I want to bring my family also, my kids will enjoy it I think.",
    ],
    "interview": [
        "Hi, thank you for having me. I am a software engineer with around three years experience in backend development.",
        "In my current role, I am responsible for designing APIs and I also mentor two junior developers.",
        "One challenge I faced was when our database was very slow, so I optimized the queries and added caching, and the performance improved a lot.",
        "My biggest strength is problem solving, but sometimes I am too perfectionist and I spend too much time on details.",
        "In five years I want to become a tech lead and maybe manage a small team.",
    ],
    "conversation": [
        "Hey! Not too much, just working a lot this week. How about you?",
        "On the weekend I usually watch movies or sometimes I go for cycling if the weather is nice.",
        "Oh I love science fiction, last week I watched a movie about time travel, it was really mind blowing.",
        "Yeah, I think technology in movies is sometimes scary because maybe one day it become real.",
        "Haha true. Anyway, I should probably get back to work soon.",
    ],
    "professional": [
        "Good morning. I'd like to give you a quick update on the project status.",
        "We have completed the first phase, but we are facing some delay in the testing because of resource constraint.",
        "My proposal is to bring one more engineer for two weeks to catch up the timeline.",
        "I understand the budget concern, but the delay will cost us more if we miss the client deadline.",
        "Sure, I will prepare a detailed cost breakdown and send it by tomorrow end of day.",
    ],
    "accent": [
        "I need to think about it.",
        "I tink the answer is dis one.",  # deliberate TH error
        "The right approach really matters.",
        "I am not sure, can you say it again?",
        "Okay, I need to think about it. Better now?",
    ],
}

GREETINGS = {
    "diagnostic": "Hi Sara, welcome — I'm Echo, your English coach. This first chat is just so I can get to know you and build you a plan. No test, no right answers — talk to me naturally. So tell me: what's bringing you to English right now, and what are you hoping to get out of it?",
    "training": "Hey Sara! Good to have you. Let's get straight into it — tell me, what's been keeping you busy lately?",
    "interview": "Thanks for coming in, Sara. Let's jump straight to it — walk me through your background. Who are you and what do you do?",
    "conversation": "Hey Sara! Great to chat. So — what's going on with you lately? Anything interesting happen this week?",
    "professional": "Right Sara, let's get started. Give me a quick summary of what you're currently working on.",
    "accent": "Welcome Sara. We're going to work on your pronunciation through shadowing. I'll say a phrase, you repeat it back. Ready? Let's start simple: \"I need to think about it.\" Your turn — repeat that.",
}


def build_system_prompt(mode: str, turn_count: int, cefr="B1", accent="american") -> str:
    """Mirror handle_stream_b's prompt assembly."""
    if mode == "diagnostic":
        return DIAGNOSTIC_CONVERSATION_SYSTEM
    curriculum_ctx = build_session_plan(
        mode=mode, cefr=cefr, accent=accent, memory={}, turn_count=turn_count, top_errors=[]
    )
    if mode == "interview":
        return curriculum_ctx + build_interview_prompt()
    if mode == "conversation":
        return curriculum_ctx + build_conversation_prompt(cefr, accent)
    if mode == "professional":
        return curriculum_ctx + build_professional_prompt(cefr, accent)
    if mode == "accent":
        phoneme_ctx = build_accent_session_context(accent, 0, turn_count)
        return phoneme_ctx + build_accent_prompt(accent, cefr, context="daily", curriculum_context="")
    return curriculum_ctx + build_training_prompt(cefr, [], "general", accent)


async def run(mode: str):
    script = LEARNER_SCRIPTS[mode]
    greeting = GREETINGS[mode]
    history = [{"role": "assistant", "content": greeting}]

    print("=" * 78)
    print(f"  SIMULATED SESSION — mode: {mode}")
    print("=" * 78)
    print(f"\n[ECHO greeting]\n  {greeting}\n")

    assessment_complete = False
    for i, user_msg in enumerate(script):
        print(f"[USER turn {i+1}]\n  {user_msg}\n")
        history.append({"role": "user", "content": user_msg})

        sysp = build_system_prompt(mode, turn_count=i + 1)
        state = {"conversation_history": list(history)}
        reply = ""
        async for tok in stream_conversation(sysp, user_msg, state):
            reply += tok

        if mode == "diagnostic" and "[ASSESSMENT_COMPLETE]" in reply:
            assessment_complete = True
            reply = reply.replace("[ASSESSMENT_COMPLETE]", "").strip()

        print(f"[ECHO turn {i+1}]  ({len(reply.split())} words)\n  {reply}\n")
        history.append({"role": "assistant", "content": reply})

        if assessment_complete:
            print(">>> Echo emitted [ASSESSMENT_COMPLETE] — ending conversation.\n")
            break

    # For diagnostic, run scoring + roadmap on the full transcript
    if mode == "diagnostic":
        transcript = "\n".join(
            f"{'Assistant' if m['role']=='assistant' else 'User'}: {m['content']}"
            for m in history
        )
        print("=" * 78)
        print("  RUNNING DIAGNOSTIC SCORING + ROADMAP")
        print("=" * 78)
        res = await stream_diagnostic_score(transcript)
        if not res:
            print("SCORING FAILED")
            return
        print(f"\nCEFR: {res.get('cefr_level')}   IELTS: {res.get('ielts_equivalent')}")
        print(f"  Fluency {res.get('fluency')} | Lexical {res.get('lexical')} | "
              f"Grammar {res.get('grammar')} | Pronunciation {res.get('pronunciation')}")
        print(f"\nStrengths:")
        for s in res.get("strengths", []): print(f"  + {s}")
        print(f"Weaknesses:")
        for w in res.get("weaknesses", []): print(f"  - {w}")
        rm = res.get("roadmap", {})
        print(f"\nGOAL: {rm.get('goal')}")
        print(f"MOTIVATION: {rm.get('motivation')}")
        print(f"TIMELINE: {rm.get('timeline')}")
        print(f"\nLEARNING PLAN: {res.get('learning_plan')}")
        print(f"\nROADMAP ({len(rm.get('sessions', []))} sessions):")
        for s in rm.get("sessions", []):
            print(f"  {s['n']}. [{s['mode']}] {s['title']}  (~{s['duration_min']}min)")
            print(f"     focus: {s['focus']}")
            print(f"     {s['description']}")
            if s.get("skills"): print(f"     skills: {', '.join(s['skills'])}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diagnostic"
    asyncio.run(run(mode))
