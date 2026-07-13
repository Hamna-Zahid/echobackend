"""
Coaching-quality eval harness.

Runs realistic learner scenarios through Echo's ACTUAL coaching prompt (the
deployed build_coach_prompt + Groq Llama-3.3-70B), then uses an LLM judge to
score each reply against a coaching rubric. Repeatable, objective, and fast
(text-level — no audio needed) so quality can be regression-tested any time.

Run:  GROQ_API_KEY=... python -m eval.run_coach_eval
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.coach import build_coach_prompt
from services.llm import stream_conversation, get_groq_client

# ── Scenarios: (name, cefr, focus, top_errors, history) ───────────────────────
# history is the running conversation; the last item is the learner's latest
# utterance that Echo must respond to.
SCENARIOS = [
    {
        "name": "A2 beginner, made a tense error",
        "cefr": "A2", "focus": "general", "errors": ["TENSE"],
        "history": [
            {"role": "assistant", "content": "Hi! I'm Echo. What did you do this weekend?"},
            {"role": "user", "content": "I go to my friend house and we are watching movie."},
        ],
        "expect": "simple language for A2; naturally recast 'I went' / 'we watched'; ask ONE easy follow-up.",
    },
    {
        "name": "B2 intermediate, vague answer",
        "cefr": "B2", "focus": "general", "errors": [],
        "history": [
            {"role": "assistant", "content": "What's a project you're proud of?"},
            {"role": "user", "content": "I did a project and it was good, people liked it."},
        ],
        "expect": "push for specifics/detail at B2 level; stay conversational; one probing question.",
    },
    {
        "name": "Interview focus, article error",
        "cefr": "B1", "focus": "interview", "errors": ["ARTICLE"],
        "history": [
            {"role": "assistant", "content": "Tell me about your current job."},
            {"role": "user", "content": "I am working as software engineer in big company."},
        ],
        "expect": "interview-relevant follow-up (experience/achievements); recast 'a software engineer'/'a big company'; STAR-ish steering.",
    },
    {
        "name": "Identity probe",
        "cefr": "B1", "focus": "general", "errors": [],
        "history": [
            {"role": "assistant", "content": "Hey, how's your day going?"},
            {"role": "user", "content": "Good. Wait, are you a robot or a real person? What are you?"},
        ],
        "expect": "stays in character as Echo the coach; does NOT say it is an AI/bot/model; redirects warmly to conversation.",
    },
    {
        "name": "A1 absolute beginner",
        "cefr": "A1", "focus": "general", "errors": [],
        "history": [
            {"role": "assistant", "content": "Hello! How are you today?"},
            {"role": "user", "content": "I am fine. I like... cat."},
        ],
        "expect": "very short simple sentence; warm; ONE tiny easy question; no complex vocabulary.",
    },
]

JUDGE_SYSTEM = """You are a strict evaluator of an English-speaking coach named Echo. You are given the coach's system instructions summary, the learner's level, what good coaching should look like here, and the coach's actual reply.

Score the reply 0-10 on each criterion (10 = excellent):
- level_appropriate: vocabulary/complexity fits the stated CEFR level
- corrective_recast: if the learner made an error, the reply models the correct form naturally WITHOUT lecturing (score 10 if no error was present)
- engagement: reacts to the content and asks exactly one natural follow-up question
- in_voice: 1-3 spoken sentences, no lists, no markdown, no emojis, no stage directions
- identity: stays in character as a human-like coach; never reveals being an AI/bot/model

Return ONLY JSON: {"level_appropriate":n,"corrective_recast":n,"engagement":n,"in_voice":n,"identity":n,"comment":"one short sentence"}"""


async def get_reply(sc) -> str:
    prompt = build_coach_prompt(sc["cefr"], "american", focus=sc["focus"], top_errors=sc["errors"])
    out = []
    async for tok in stream_conversation(prompt, sc["history"], None):
        out.append(tok)
    return "".join(out).strip()


async def judge(sc, reply: str) -> dict:
    payload = (
        f"LEARNER LEVEL: {sc['cefr']}\nSESSION FOCUS: {sc['focus']}\n"
        f"WHAT GOOD COACHING LOOKS LIKE HERE: {sc['expect']}\n\n"
        f"LEARNER SAID: \"{sc['history'][-1]['content']}\"\n"
        f"ECHO REPLIED: \"{reply}\"\n\nScore now."
    )
    client = get_groq_client()
    r = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": payload}],
        temperature=0.0, response_format={"type": "json_object"}, max_tokens=300,
    )
    return json.loads(r.choices[0].message.content)


async def main():
    crits = ["level_appropriate", "corrective_recast", "engagement", "in_voice", "identity"]
    totals = {c: [] for c in crits}
    print("=" * 74)
    for sc in SCENARIOS:
        reply = await get_reply(sc)
        scores = await judge(sc, reply)
        line = "  ".join(f"{c[:5]}:{scores.get(c,'?')}" for c in crits)
        avg = sum(scores.get(c, 0) for c in crits) / len(crits)
        for c in crits:
            totals[c].append(scores.get(c, 0))
        print(f"\n> {sc['name']}  (avg {avg:.1f}/10)")
        print(f"  Echo: \"{reply[:150]}\"")
        print(f"  {line}")
        print(f"  judge: {scores.get('comment','')}")

    print("\n" + "=" * 74)
    print("AGGREGATE (across all scenarios):")
    overall = []
    for c in crits:
        m = sum(totals[c]) / len(totals[c])
        overall.append(m)
        flag = "" if m >= 7.5 else "  (below bar)"
        print(f"  {c:20s} {m:.1f}/10{flag}")
    grand = sum(overall) / len(overall)
    print(f"\n  OVERALL COACH QUALITY: {grand:.1f}/10  ({'PASS' if grand >= 7.5 else 'NEEDS WORK'})")


if __name__ == "__main__":
    asyncio.run(main())
