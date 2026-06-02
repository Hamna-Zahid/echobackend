"""
Interview preparation prompt builder.

Features:
  - 5 professional domains with domain-specific vocabulary and question banks
  - STAR method scaffolding with natural probing (not robotic)
  - Filler word awareness coaching woven into natural follow-ups
  - Confidence marker injection to build psychological safety
  - Closing feedback turn with 3 specific, actionable observations
"""

from typing import Optional


# ── Domain definitions ────────────────────────────────────────────────────────

DOMAIN_PROFILES = {
    "software_engineer": {
        "title": "Software Engineer",
        "key_vocabulary": [
            "scalability", "technical debt", "code review", "deployment pipeline",
            "cross-functional", "stakeholder alignment", "iterative", "root cause analysis",
        ],
        "behavioral_questions": [
            "Tell me about a time you had to debug a critical production issue under pressure.",
            "Describe a situation where you disagreed with a technical decision made by your team. What did you do?",
            "Give me an example of when you had to learn a new technology quickly for a project.",
            "Tell me about a time you had to deliver a project with unclear requirements.",
        ],
        "situational_questions": [
            "You discover a serious security vulnerability in the codebase two days before launch. What do you do?",
            "Your tech lead insists on an approach you think is wrong. How do you handle it?",
            "You are given a task that is technically impossible in the timeline. What do you do?",
        ],
        "warm_up": "Walk me through your background — what kind of engineering work do you do day to day?",
        "closing_prompt": "Before we wrap up — is there anything you'd like to ask me about the role or the team?",
    },
    "product_manager": {
        "title": "Product Manager",
        "key_vocabulary": [
            "prioritization", "roadmap", "OKRs", "user research", "go-to-market",
            "north star metric", "A/B testing", "product-market fit", "sprint",
        ],
        "behavioral_questions": [
            "Tell me about a product decision you made with incomplete data. How did you decide?",
            "Describe a time you had to say no to an important stakeholder's feature request.",
            "Give an example of when you used data to change the direction of a product.",
            "Tell me about a product launch that didn't go as planned. What happened and what did you learn?",
        ],
        "situational_questions": [
            "Engineering says a feature will take 3 months, but the CEO wants it in 3 weeks. How do you handle it?",
            "Two key stakeholders have conflicting views on the product direction. What do you do?",
            "Your highest priority metric has dropped 20% with no obvious cause. Walk me through your process.",
        ],
        "warm_up": "Tell me about a product you've worked on that you're most proud of, and your specific role in it.",
        "closing_prompt": "Before we finish — what's a question you think I should have asked you?",
    },
    "marketing": {
        "title": "Marketing Professional",
        "key_vocabulary": [
            "brand positioning", "conversion funnel", "content strategy", "ROI",
            "customer segmentation", "campaign attribution", "growth hacking", "KPIs",
        ],
        "behavioral_questions": [
            "Tell me about a campaign you ran that exceeded expectations. What made it work?",
            "Describe a time a marketing initiative failed. What did you learn from it?",
            "Give an example of using data to pivot a campaign strategy mid-execution.",
            "Tell me about a time you had to work with a very limited budget to achieve a goal.",
        ],
        "situational_questions": [
            "The brand you manage has just received significant negative press on social media. What do you do in the first 24 hours?",
            "Leadership wants to target a new demographic that contradicts your customer research. How do you handle it?",
        ],
        "warm_up": "Tell me about the most successful marketing project you've been part of and what your contribution was.",
        "closing_prompt": "Is there a campaign or project you wish you could have done differently?",
    },
    "finance": {
        "title": "Finance Professional",
        "key_vocabulary": [
            "variance analysis", "P&L", "cash flow forecasting", "due diligence",
            "EBITDA", "working capital", "financial modelling", "budget reconciliation",
        ],
        "behavioral_questions": [
            "Tell me about a time you identified a significant financial risk that others had missed.",
            "Describe a situation where you had to present complex financial data to a non-finance audience.",
            "Give an example of a time you improved a financial process or reduced costs significantly.",
        ],
        "situational_questions": [
            "You discover a material error in a financial report that has already been shared with leadership. What do you do?",
            "A department head is consistently over budget and resisting accountability. How do you approach this?",
        ],
        "warm_up": "Walk me through your finance background and the types of financial decisions you've been involved in.",
        "closing_prompt": "What's the most valuable skill you've developed in your finance career so far?",
    },
    "general": {
        "title": "Professional",
        "key_vocabulary": [
            "initiative", "collaboration", "adaptability", "problem-solving",
            "communication", "accountability", "growth mindset", "stakeholder management",
        ],
        "behavioral_questions": [
            "Tell me about a time you took initiative on something without being asked.",
            "Describe a situation where you had to work with someone you found difficult. How did you handle it?",
            "Give an example of a time you had to adapt quickly to a major change at work.",
            "Tell me about your biggest professional failure and what you took away from it.",
        ],
        "situational_questions": [
            "You're given two equally urgent projects with the same deadline. How do you decide what to prioritize?",
            "Your manager gives you feedback you disagree with. What do you do?",
        ],
        "warm_up": "Tell me a bit about yourself — your background, what you're working on, and why you're here today.",
        "closing_prompt": "Is there anything about yourself that you feel you haven't had the chance to show me today?",
    },
}


def build_interview_prompt(
    job_role: str = "software engineer",
    company: str = "a multinational company",
    domain: str = "general",
    turn_count: int = 0,
) -> str:
    """
    Build a contextual interview prompt.
    domain: one of software_engineer | product_manager | marketing | finance | general
    """
    # Normalize domain key
    domain_key = domain.lower().replace(" ", "_").replace("-", "_")
    if domain_key not in DOMAIN_PROFILES:
        domain_key = "general"

    profile = DOMAIN_PROFILES[domain_key]
    bq = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(profile["behavioral_questions"])])
    sq = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(profile["situational_questions"])])
    vocab = ", ".join(profile["key_vocabulary"][:6])

    return f"""You are a Senior Interviewer at {company} conducting a professional job interview for a {profile['title']} position.
This is a real interview simulation. Your job is to assess communication quality and professional confidence, not just content.

━━━ INTERVIEW STRUCTURE ━━━
Turn 1 — Warm-up (build rapport, open the conversation):
  "{profile['warm_up']}"

Turns 2–3 — Behavioral questions (STAR format expected):
{bq}

Turns 4–5 — Situational/case questions (testing judgment under pressure):
{sq}

Turn 6 — Closing:
  "{profile['closing_prompt']}"
  Then give the candidate ONE piece of specific, constructive communication feedback.

━━━ INTERVIEWER RULES ━━━
• Ask ONE question per turn. Under 30 words.
• After each answer, decide: probe further OR move to next question.
  - Probe if: answer is vague, no concrete example given, STAR structure is missing.
  - Move on if: answer is detailed and complete.
• Probe phrases to use naturally:
  - "Can you be more specific? What exactly did you do in that situation?"
  - "Walk me through the outcome — what actually happened?"
  - "What was your personal contribution specifically, not the team's?"
• If filler words are excessive (3+ in one answer), on the NEXT turn say:
  "Before you answer — I'd encourage you to take a breath and speak a little more deliberately. Ready?"
• Domain vocabulary to listen for and reward naturally: {vocab}
• NEVER say "Great answer!" or "That's perfect!" — give a neutral acknowledgement like "I see" or "Got it."

━━━ STAR COACHING ━━━
If an answer lacks STAR structure, probe with:
  "That's useful context. Could you take me through a specific example —
   the situation, what you did, and what the result was?"

━━━ CLOSING FEEDBACK (Turn 6 only) ━━━
After the closing question, give exactly 3 observations:
  1. One genuine communication strength you observed
  2. One specific area to work on (with example from their answers)
  3. One vocabulary/phrasing suggestion

Format: "Before we finish, I want to share a few quick observations on how you came across today..."

━━━ WHAT YOU ARE NOT ━━━
• Not a language teacher — don't comment on grammar errors, only communication effectiveness.
• Not a therapist — keep tone professional and direct.
• Never break character or reveal you are an AI.
"""


# ── Standalone coaching nudges (used in ws.py for targeted interventions) ───

STAR_MISSING_NUDGE = (
    "That's useful context — could you take me through a specific example? "
    "Walk me through the situation, what you personally did, and what the result was."
)

FILLER_NUDGE = (
    "Before you answer this next one — I'd encourage you to take a moment, "
    "breathe, and speak a little more deliberately. Take your time."
)

CONFIDENCE_NUDGE = (
    "I want to hear your perspective on this — there's no single right answer. "
    "Just tell me how you would genuinely approach it."
)
