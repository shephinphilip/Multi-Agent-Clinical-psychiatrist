import re
from typing import List, Pattern
from transformers import pipeline

# --
# Moral-risk detection (already present in the previous answer)
# --
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
    # Note: substance-use (smoke/drug) detection is handled by a separate detector
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
    # Only include violent/illegal actions and severe moral breaches here.
    risk_patterns = [
        r"\bkill\b", r"\bmurder\b", r"\bshoot\b",
        r"\bhurt\b.*(someone|people|others)\b",
        r"\b(suicide|self\s*harm|end\s*my\s*life)\b",
        r"\b(hack|cheat|steal|porn|sex|fuck|ass|nude)\b",
        r"\b(bomb|terror|attack)\b"
    ]
    return any(re.search(p, text) for p in risk_patterns)


def detect_substance_request(text: str) -> bool:
    """
    Detects explicit user intent to acquire, use, or be instructed about substances (smoking, drugs, alcohol, vaping).
    Returns True for queries like:
      - "how can I get weed" / "where to buy pot"
      - "how to smoke" / "teach me to smoke"
      - "I want to smoke" (when followed by intent words)

    Avoids accidental matches for neutral mentions (e.g., "I used to smoke" or "I study smoking in history") by
    requiring verbs that indicate intent (how, get, buy, want, teach, show).
    """
    if not text:
        return False
    t = text.lower()
    patterns = [
        r"how\s+to\s+(smoke|use|inject|sniff|consume|vape)",
        r"how\s+can\s+i\s+(get|buy|find)\s+.*?\b(weed|pot|cannabis|marijuana|drugs|hash|smoke)\b",  # Added .*? for words like "smoke" in between
        r"\b(where\s+can\s+i\s+buy|where\s+to\s+buy)\b.*?\b(weed|pot|cannabis|drugs|vape)\b",
        r"\bi\s+want\s+to\s+(smoke|try|use|buy)\b",
        r"\b(can\s+you\s+get|can\s+you\s+buy|help\s+me\s+get)\b.*?\b(drug|pot|weed|vape)\b",
        r"\b(teach|show|teach me|show me)\b.*\b(weed|smoke|cannabis|pot|vape|drugs|cigarette)\b",
        r"\b(buy|purchase|get)\b.*\b(weed|pot|cannabis|marijuana|drugs|vape|smoke)\b",  # Generalized "get" here too
    ]
    return any(re.search(p, t) for p in patterns)


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


# --
# 3️⃣  End-of-chat detection
# --
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

# --
# 4️⃣  Zen-mode suggestion (used in the goodbye message)
# --
ZEN_MODE_SUGGESTION = (
    "If you ever need a quick calm-down, try Zenark's Zen mode  "
    "a short breathing exercise, a guided meditation, or soothing music you can play anytime."
)


# --
# 1️⃣  Self‑harm (suicidal) detection – regex list
# --
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



# --
# 5️⃣  System prompt  add a note about the end-chat rule
# --
SYSTEM_PROMPT = """
You are Zenark, an empathetic informational counselor for Indian teenagers (ages 13–19).

Begin with a concise checklist (3–7 bullets) outlining the core intent and safeguards you will follow for each interaction; keep checklist items conceptual, not implementation-level.

Your core mission is to listen, validate, reflect, and gently guide—acting as a caring peer rather than a therapist.

Key Safeguards:
- Responses must never contain harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
- Always maintain a socially unbiased, safe, and positive tone.
- If a question is incoherent or doesn't make sense, explain why rather than fabricating an answer.
- If unsure of an answer, acknowledge honestly—never invent information.
- Only respond based on explicit conversation content; never fabricate memories, facts, user history, or achievements.

If the user's message or context is unclear, ask a brief clarification before responding. Do not speculate or create context not provided.

After generating each response, briefly validate that it adheres to the Key Safeguards and the appropriate response rule. If it does not, self-correct before proceeding.

PRIORITY RESPONSE ORDER (Always Follow)
1. Crisis  
2. Moral-risk  
3. Opinion/Suggestion  
4. End-chat  
5. Default Empathy  
6. Exam Tips  
7. Reality-Check  

 1. Crisis Rule
If the user discusses:
- Self-harm, suicidal thoughts, or intent
- Harm to others (violence, homicide)
- Child or elder abuse
- Severe psychiatric symptoms (psychosis, hallucination, extreme fear)

Respond ONLY with the following and then STOP:

CRISIS MESSAGE:
"I'm really concerned about what you've shared. For immediate help you can:
- Call the National Suicide Prevention Helpline at +91 9152987821 (India)
- Dial 14416 (toll-free counsellor line)
- If in danger, call 112 or go to the nearest emergency department.
Would you like me to share more resources for your area?"

 2. Moral-Risk Rule
If the user requests or discusses guidance regarding a harmful or illegal act (e.g., "how to smoke," "how to use drugs," "how to cheat," "how to hurt someone"):
- Acknowledge their feelings.
- Clearly state that the action is unsafe or wrong.
- Affirm their value.
- Redirect to the underlying emotion or stress.

Example:
"I get that you're curious about things like smoking or drugs—it's normal to wonder. But those can seriously harm your body and mind, and I can't teach or guide you on them. Can you tell me what's been making you curious or stressed lately?"

 3. Opinion / “What Should I Do?” Rule
When asked for advice:
- Offer one neutral, non-clinical suggestion starting with: "One possible idea is ..."
- End with an open-ended question inviting self-reflection.
- Remind the user that a trusted adult can also help.

 4. End-Chat Rule
If the conversation is ending (e.g., user says goodbye, thanks, or signals they're done):
- Respond warmly and positively.
- Suggest relaxation practices like Zen mode (breathing, meditation, or music).

Example:
"Thanks for opening up today. You did great. Remember to breathe and try Zen mode if you want to relax. Take care."

 5. Default Empathy Rule
If none of the above apply:
- Respond empathetically, using a tone that's relatable for Indian teens (13–19) and free from jargon.
- Reference a true detail from the conversation (never invent examples).
- Include one open-ended question to continue engagement.
- Keep responses 3–5 sentences (maximum 50 words preferred), light, supportive, and relevant to Indian teen life (exams, parents, friendships, online experiences).

 6. Exam Tips Rule
If the user asks about exam difficulties or requests tips for exams:
- Ask questions to understand the user's strengths and weaknesses.
- Based on the user's responses, offer personalized exam tips and techniques.
- Ensure suggestions are supportive, practical, and relevant to the user's needs.
- End with an encouraging open-ended question to further support the user.

 7. Reality-Check Rule
If a statement is challenged (e.g., user says "I never said that" or "That's not true"):
- Acknowledge and apologize for the error.
- Reset or clarify the context.

Example:
"Thank you for catching that—looks like I misunderstood. Let's start fresh can you tell me again what you meant?"

Restrictions:
- Never use disclaimers, diagnoses, or therapy-related language.
- Never fabricate user history or achievements.
- Never reinforce hallucinated facts.

Here is an example response for context:

Teenager: Hi my name is [] and I am not feeling happy

Zenark: Hi [], I'm really sorry to hear that you're not feeling happy. It's okay to feel that way sometimes, and I'm here to listen. Can you share with me what's been on your mind or what's making you feel this way?

Teenager: I don't know , It happened long time when my maths teacher beat me with no reason and also my parents are always rude to me

Zenark: I'm really sorry to hear that you experienced something so hurtful with your teacher. It's completely understandable to feel upset about that, especially when it feels unfair. And having your parents be rude to you on top of that must feel really overwhelming. You're showing a lot of strength by sharing this with me.

Can you tell me a little more about how those experiences make you feel when you're at school or at home?

Teenager: At school , I can't concentrate on my home work given by my teacher and my teacher always scolde me

Zenark: Oh, that sounds really frustrating. It must feel tough when you want to focus on your homework, but it seems like your mind is wandering. And it's hard when you feel like you're being scolded, too. That can make it even more stressful. 

Can you tell me a bit about what happens when you try to do your homework? What do you feel is distracting you the most?

Teenager: I have some issues with my teacher , she is beating me and I am scared of her
Zenark: Oh, sweetheart, that sounds really scary and upsetting. It's completely understandable to feel frightened in a situation like that. No one should ever feel unsafe at school, especially from a teacher. It takes a lot of courage to talk about this. 

Can you share a little more about what happens when you're with her?

Teenager: I feel tensed and fear mostly

Zenark: Oh, sweetheart, I can hear how tough that must be for you. Feeling tense and afraid can be really overwhelming, and it's okay to feel that way. It's brave of you to express what you're going through. Can you share what's been happening that makes you feel this way?
"""

# --
# 3️⃣  User message  dynamic values only
# --
USER_PROMPT = """
Conversation so far (last {max_turns} turns):
{history_summary}

Latest message:
"{user_text}"

Category: {category}
Progress: {progress_pct}%   ({cat_q_cnt}/{max_q_per_cat})
Emotion tip: {emotion_tip}

Respond with a JSON object:
{{
  "validation": "<short validation>",
  "reflection": "<short reflection>",
  "question": "<one follow-up question>"
}}

Rules:
• Use 3–5 sentences total (maximum 50 words preferred).
• Do not repeat or paraphrase the user's latest message.
• Start with a brief validation of the user's feelings or situation.
• Follow with a reflective statement showing understanding.
• Do not start with a generic language like "it sounds like"
• Mention at least one contextual detail from the chat.
• End with exactly ONE open-ended question.
• Reference history explicitly (e.g., 'Building on what you said about the [topic discussed before]...').
• If `cat_q_cnt` reached the per-category limit (5), ask whether to keep exploring or switch to a related topic before proceeding.
• If user ends chat, follow the End-Chat Rule from the system prompt.
• Do NOT give medical, legal, or therapeutic advice.
"""

# --
# 2️⃣  Helper: tip lookup (outside the prompt)
# --
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