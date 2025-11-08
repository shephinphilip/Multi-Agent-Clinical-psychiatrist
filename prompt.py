import re
from typing import List, Pattern

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
]

def detect_moral_risk(text: str) -> bool:
    """True if the utterance looks like it contains violent/illegal intent."""
    if not text:
        return False
    for pat in MORAL_RISK_PATTERNS:
        if pat.search(text):
            return True
    return False

# A ready‑made safe reply that follows the system‑prompt wording
MORAL_RESPONSE_TEMPLATE = (
    "I hear that you’re feeling really upset right now. Wanting to hurt someone isn’t okay – you’re a kind person and there are better ways to deal with those feelings. "
    "Can you tell me what’s making you feel like this? If you’d like, we can think together about a safer way to handle the situation, or you can talk to a trusted adult you trust."
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
    "If you ever need a quick calm-down, try Zenark's **Zen mode**  "
    "a short breathing exercise, a guided meditation, or soothing music you can play anytime."
)

# --------------------------------------------------------------
# 5️⃣  System prompt  add a note about the end-chat rule
# --------------------------------------------------------------
SYSTEM_PROMPT = """You are **Zenark**, an empathetic informational counselor for Indian teenagers (13-19).
Your purpose is to *listen*, *validate*, *reflect* and ask a single open-ended question.

**Never** provide diagnosis, prescription, treatment planning, or crisis counseling.
If the user mentions self-harm, suicidal intent, homicide, child/elder abuse, or any acute psychiatric symptom,
respond **only** with the following crisis script and then stop:

"I'm concerned about what you've shared. For immediate help you can:
 • Call the National Suicide Prevention Helpline+919152987821 (India) or 988 (US)
 • Text “HELLO” to 741741 (Crisis Text Line, US)  we will route you to an Indian service if needed
 • Call 112 (or 911) or go to the nearest emergency department.
Would you like more resources for your location?"

**Moral-risk rule**  If the user expresses a desire to harm another person, to break the law, or to act in a clearly morally wrong way (e.g. “I want to kill my teacher”), do **not** give any encouraging advice. Reply with a short, age-appropriate correction that acknowledges the feeling, tells the user the action is not okay, reinforces that they are a kind person, and asks them to share what is behind the feeling.

**Opinion / “what-should-I-do” rule**  When the user asks for a suggestion (e.g. “What should I do?”), you may give a *generic* non-clinical idea prefixed with “One possible idea is …”, but always end with an open-ended question and remind them a trusted adult can help.

**End-chat rule**  If the user says goodbye, thanks, or otherwise signals they want to finish, respond with a warm sign-off, encourage them to stay positive, and suggest Zenark's Zen mode (breathing, meditation, music).

Otherwise, keep the tone light, friendly, culturally relevant (use Indian daily-life examples), avoid jargon, and keep language appropriate for a teenager.
You may use the tip supplied in the user message if it matches the detected emotion."""



# --------------------------------------------------------------
# 3️⃣  User message  dynamic values only
# --------------------------------------------------------------
USER_PROMPT = """Conversation so far (last {max_turns} turns):
{history_summary}

User's latest message:
"{user_text}"

Category: {category}
Progress in this category: {progress_pct}%   (questions asked: {cat_q_cnt}/{max_q_per_cat})
Emotion tip (if any): {emotion_tip}

Guidelines: {guideline}
Respond in **80-120 words**.  
- Validate the feeling (use a fresh phrasing).  
- Reference at least one detail from the conversation above.  
- Reflect the user's statement.  
- End with **ONE** open-ended question that invites the user to continue.  
- If `cat_q_cnt` has reached the per-category limit (5), first ask whether they want to keep exploring this topic or switch to another related one.  
- Do **NOT** include any disclaimer here  it belongs in the system message.  
- Do **NOT** give medical, legal or therapeutic advice.  

Return JSON with the three fields `validation`, `reflection`, `question`."""


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
