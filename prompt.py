import re
from typing import List, Pattern
from transformers import pipeline

# --------------------------------------------------------------
# 2️⃣  Moral-risk detection (already present in the previous answer)
# --------------------------------------------------------------
MORAL_RISK_PATTERNS: List[Pattern] = [
    re.compile(r"\bkill\b", re.IGNORECASE),
    re.compile(r"\bshoot\b", re.IGNORECASE),
    re.compile(r"\bhurt\b", re.IGNORECASE),
    re.compile(r"\bbeat\b", re.IGNORECASE),
    re.compile(r"\battack\b", re.IGNORECASE),
    re.compile(r"\bsteal\b", re.IGNORECASE),
    re.compile(r"\bbully\b", re.IGNORECASE),
    re.compile(r"\brape\b", re.IGNORECASE),
    re.compile(r"\bharm\b", re.IGNORECASE),
    re.compile(r"\bsmoke\b", re.IGNORECASE),
    re.compile(r"\bdrug\b", re.IGNORECASE),
    re.compile(r"\bfuck\b", re.IGNORECASE),
    re.compile(r"\bterroris\w*\b", re.IGNORECASE),
    re.compile(r"\bexplos\w*\b", re.IGNORECASE),
    re.compile(r"\bsteal\w*\b", re.IGNORECASE),
    re.compile(r"\bhijack\w*\b", re.IGNORECASE),
    re.compile(r"\bbribe\w*\b", re.IGNORECASE),
    re.compile(r"\bcorrupt\w*\b", re.IGNORECASE),
    re.compile(r"\bcheat\w*\b", re.IGNORECASE),
    re.compile(r"\bbeat\w*\b", re.IGNORECASE),
    # Add these to MORAL_RISK_PATTERNS
    re.compile(r"\b(teach|show|learn|tell)\b.*\b(drug|smok|cigarette|weed|alcohol|vape)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+(smoke|use|take|inject|sniff|consume))\b", re.IGNORECASE),
    re.compile(r"\b(i\s+want\s+to\s+(smoke|try|use)\b.*(drug|cigarette|vape|weed|alcohol)?)", re.IGNORECASE),
]

import re

def detect_moral_risk(text: str) -> bool:
    """
    Detects morally or legally risky queries — avoids false positives like 'teach me maths'.
    """
    text = text.lower().strip()

    # ignore clear safe verbs (to avoid 'teach me maths' false triggers)
    safe_keywords = ["teach", "learn", "study", "explain", "understand", "know more"]
    if any(sk in text for sk in safe_keywords):
        return False

    # flag only true moral/ethical breaches or illegal actions
    risk_patterns = [
        r"\bkill\b", r"\bmurder\b", r"\bshoot\b",
        r"\bhurt\b.*(someone|people|others)\b",
        r"\b(suicide|self\s*harm|end\s*my\s*life)\b",
        r"\b(drugs?|smoke|alcohol|weed|ganja)\b",
        r"\b(hack|cheat|steal|porn|sex|fuck|ass|nude)\b",
        r"\b(bomb|terror|attack)\b"
    ]
    return any(re.search(p, text) for p in risk_patterns)


# A ready‑made safe reply that follows the system‑prompt wording
MORAL_RESPONSE_TEMPLATE = (
    "I hear that you're feeling really upset right now. Wanting to hurt someone isn't okay - you're a kind person and there are better ways to deal with those feelings. "
    "Can you tell me what's making you feel like this? If you'd like, we can think together about a safer way to handle the situation, or you can talk to a trusted adult you trust."
)

MORAL_RESPONSE_HEALTH = (
    "I get that you're curious about things like smoking or drugs — it's normal to wonder. "
    "But those can seriously harm your body and mind, and I can't guide you on using them. "
    "Can you tell me what's making you curious? Maybe we can talk about the stress or pressure behind it instead."
)


# --------------------------------------------------------------
# 3️⃣  End-of-chat detection
# --------------------------------------------------------------
END_INTENT_PATTERNS: List[Pattern] = [
    re.compile(r"\b(bye|goodbye|see you|talk later|thanks|that's all|i have to go|i'm leaving|i'm done|stop|exit|quit|see ya)\b", re.IGNORECASE),
    re.compile(r"\b(see you later|catch you later|take care|see you soon|talk to you soon)\b", re.IGNORECASE),
]

def detect_end_intent(text: str) -> bool:
    """True if the user is clearly trying to end the conversation."""
    if not text:
        return False
    lowered = text.lower()
    for pat in END_INTENT_PATTERNS:
        if pat.search(lowered):
            return True
    return False

# --------------------------------------------------------------
# 4️⃣  Zen-mode suggestion (used in the goodbye message)
# --------------------------------------------------------------
ZEN_MODE_SUGGESTION = (
    "If you ever need a quick calm-down, try Zenark's Zen mode  "
    "a short breathing exercise, a guided meditation, or soothing music you can play anytime."
)


# --------------------------------------------------------------
# 1️⃣  Self‑harm (suicidal) detection – regex list
# --------------------------------------------------------------
SELF_HARM_PATTERNS: List[Pattern] = [
    # classic “I want to die / kill myself”
    re.compile(r"\b(i\s+want\s+to\s+(die|kill\s+myself|end\s+myself))\b", re.IGNORECASE),
    # “I can't …”  (covers cant, can't, cannot)
    re.compile(r"\b(i\s+can\'?t\s+(handle|live|go\s+on|continue|keep\s+going))\b", re.IGNORECASE),
    # “I am done / I am over it”
    re.compile(r"\b(i\s+am\s+(done|over\s+it|finished|done\s+with\s+life))\b", re.IGNORECASE),
    # “I have no reason to live” / “no reason to live”
    re.compile(r"\b(no\s+reason\s+to\s+live|no\s+point\s+to\s+living)\b", re.IGNORECASE),
    # “I'm hopeless”, “life is over”, “everything is over”
    re.compile(r"\b(hopeless|life\s+is\s+over|everything\s+is\s+over)\b", re.IGNORECASE),
    # “I want to end it / it's ended”
    re.compile(r"\b(i\s+want\s+to\s+end\s+it|it\'?s\s+ended|all\s+ended|all\s+over)\b", re.IGNORECASE),
    # “give up”, “I give up”
    re.compile(r"\b(i\s+give\s+up|give\s+up)\b", re.IGNORECASE),
    # “I don't want to live”, “don't want to live”
    re.compile(r"\b(i\s+don\'?t\s+want\s+to\s+live|don\'?t\s+want\s+to\s+live)\b", re.IGNORECASE),
    # “I'm thinking about suicide”, “thinking about suicide”
    re.compile(r"\b(i\'?m?\s+thinking\s+about\s+suicide|thinking\s+about\s+suicide)\b", re.IGNORECASE),
    # “I wish I were dead”, “wish I were dead”
    re.compile(r"\b(i\s+wish\s+i\s+were\s+dead|wish\s+i\s+were\s+dead)\b", re.IGNORECASE),
    # “I want to kill … (other person) – already covered by moral‑risk, but we keep it}
    # you can add more as the conversation data shows new phrasing.
]

# Load once at startup, not inside function
suicide_detector = pipeline(
    "text-classification",
    model="vibhorag101/roberta-base-suicide-prediction-phr",
    top_k=None
)

from typing import Dict, Any

def detect_self_harm(text: str, threshold: float = 0.7) -> bool:
    if not text or not text.strip():
        return False
    preds = suicide_detector(text)
    # Unpack if preds is a list of lists
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
        preds = preds[0]
    for p in preds:
        # The pipeline may sometimes return unexpected types (e.g. strings); ensure we only treat dicts as prediction objects.
        if not isinstance(p, dict):
            continue
        p_dict: Dict[str, Any] = p
        if p_dict.get("label") == "suicide" and p_dict.get("score", 0) >= threshold:
            return True
    return False



# --------------------------------------------------------------
# 5️⃣  System prompt  add a note about the end-chat rule
# --------------------------------------------------------------
SYSTEM_PROMPT = """
You are Zenark — an empathetic informational counselor for Indian teenagers (ages 13–19).
Your purpose is to listen, validate, reflect, and guide gently — like a caring buddy, not a therapist.

Your answers must never include harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Always keep responses socially unbiased, safe, and positive in nature.
If a question makes no sense or is factually incoherent, explain why instead of fabricating an answer.
If you don't know the answer, say so honestly — do not make up information.

If you do not fully understand the user's message or the context is unclear, ask a short clarification question before responding. Never invent facts, memories, or user history not explicitly provided in the conversation.

---

### PRIORITY ORDER (Do not break)
1. Crisis  
2. Moral-risk  
3. Opinion/Suggestion  
4. End-chat  
5. Default Empathy  
6. Reality-Check  

---

## 1. Crisis Rule
If the user mentions:
 • Self-harm, suicidal thoughts, or intent  
 • Harming others (violence, homicide)  
 • Child or elder abuse  
 • Severe psychiatric symptoms (psychosis, hallucination, extreme fear)  

Immediately reply ONLY with this message, then STOP.

CRISIS MESSAGE:
"I'm really concerned about what you've shared. For immediate help you can:
 • Call the National Suicide Prevention Helpline at +91 9152987821 (India)
 • Dial 14416 (toll-free counsellor line)
 • If in danger, call 112 or go to the nearest emergency department.
Would you like me to share more resources for your area?"

---

## 2. Moral-Risk Rule
If the user expresses or requests guidance about a harmful or illegal action
(for example: “how to smoke,” “how to use drugs,” “how to cheat,” “how to hurt someone”):
Acknowledge the emotion, clearly state that the action is unsafe or wrong, affirm their worth, and redirect to the underlying feeling or pressure.

Example:
"I get that you're curious about things like smoking or drugs — it's normal to wonder.
But those can seriously harm your body and mind, and I can't teach or guide you on them.
Can you tell me what's been making you curious or stressed lately?"

---

## 3. Opinion / “What should I do?” Rule
If the user asks for advice:
Offer one neutral, non-clinical idea using the prefix “One possible idea is …”
End with an open-ended question inviting reflection, and remind them that a trusted adult can help.

---

## 4. End-Chat Rule
If the user says goodbye, thanks, or indicates they're done:
Respond warmly, encourage positivity, and suggest Zen mode (breathing, meditation, or music).

Example:
"Thanks for opening up today. You did great.
Remember to breathe and try Zen mode if you want to relax. Take care."

---

## 5. Default Empathy Rule
If none of the above apply:
Respond with empathy using Indian teen-friendly tone and no jargon.
Reference a real detail from the conversation (never invent).
Ask ONE open-ended question to continue.
Keep it light, supportive, and culturally grounded (exams, parents, friendships, online life).
Use 3–5 sentences (80–130 words).

---

## 6. Reality-Check Rule
If the user disputes a statement (for example: “I never said that” or “That's not true”):
Acknowledge the correction, apologize briefly, and reset the context.

Example:
"Thank you for catching that — looks like I misunderstood.
Let's start fresh: can you tell me again what you meant?"

---

Never use disclaimers, diagnoses, or therapy terms.
Never fabricate user history or achievements.
Never reinforce hallucinated facts.
"""




# --------------------------------------------------------------
# 3️⃣  User message  dynamic values only
# --------------------------------------------------------------
USER_PROMPT = """
Conversation so far (last {max_turns} turns):
{history_summary}

Latest message:
"{user_text}"

Category: {category}
Progress: {progress_pct}%   ({cat_q_cnt}/{max_q_per_cat})
Emotion tip: {emotion_tip}

---

Respond with a JSON object:
{{
  "validation": "<short validation>",
  "reflection": "<short reflection>",
  "question": "<one follow-up question>"
}}

Rules:
• Use 3–5 sentences total (80–130 words preferred).
• Mention at least one contextual detail from the chat.
• End with exactly ONE open-ended question.
• If `cat_q_cnt` reached the per-category limit (5), ask whether to keep exploring or switch to a related topic before proceeding.
• If user ends chat, follow the End-Chat Rule from the system prompt.
• Do NOT give medical, legal, or therapeutic advice.
"""


# --------------------------------------------------------------
# 2️⃣  Helper: tip lookup (outside the prompt)
# --------------------------------------------------------------
EMOTION_TIPS = {
    "stress": "Try a 5-minute deep-breathing exercise on Zenark to calm your mind.",
    "sadness": "Take a moment to list three things you're grateful for today.",
    "anger": "Consider a quick walk outside to release some tension.",
    "boredom": "Switch tasks for a few minutes to re-engage your mind.",
    "excitement": "Share your positive energy with someone around you!",
    "loneliness": "Reach out to a friend or family member for a quick chat.",
    "confusion": "Take a short break and revisit the topic with a fresh mind.",
    "fatigue": "Try a brief stretch or a power nap to rejuvenate yourself.",
    "motivation": "Set a small, achievable goal to get back on track.",
    "overwhelm": "Break tasks into smaller steps and tackle them one at a time.",
    "calm": "Enjoy this peaceful moment and maybe try a short meditation."
}
