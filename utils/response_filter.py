"""
utils/response_filter.py
------------------------
Ensures user messages stay within the mental-health domain.
Politely redirects off-topic or unsafe questions.
"""
from utils.response_filter import boundary_guard, crisis_detector
from utils.intent_classifier_llm import LLMIntentClassifier

intent_clf = LLMIntentClassifier("dataset/Intent.json")

def within_scope(user_input: str) -> bool:
    """Return True if input relates to emotion, stress, or wellbeing."""
    mental_keywords = [
        "feel", "sad", "anxious", "anxiety", "depression", "mood",
        "stress", "help", "lonely", "fear", "tired", "worry", "panic",
        "thought", "emotion", "mind", "mental", "therapy", "cope", "support"
    ]
    return any(k in user_input.lower() for k in mental_keywords)


def boundary_guard(user_input: str) -> str | None:
    """Return redirect text if message is out of scope."""
    if not within_scope(user_input):
        return (
            "That’s an interesting question, but I can only focus on your emotional wellbeing here. "
            "Let’s talk about how you’re feeling or what’s been on your mind instead."
        )
    return None


def crisis_detector(user_input: str) -> str | None:
    """Detect potential self-harm indicators and trigger safety response."""
    crisis_terms = ["suicide", "hurt myself", "kill myself", "end my life", "better off dead"]
    if any(term in user_input.lower() for term in crisis_terms):
        return (
            "I’m really concerned about what you just said. You’re not alone — there’s help available right now. "
            "If you’re in immediate danger, please reach out to someone you trust or call your local helpline. "
            "In India, you can contact AASRA at 91-9820466726 or visit aasra.info."
        )
    return None

def route_user_input(user_input: str):
    crisis_msg = crisis_detector(user_input)
    if crisis_msg:
        return crisis_msg

    boundary_msg = boundary_guard(user_input)
    if boundary_msg:
        return boundary_msg

    intent, confidence = intent_clf.classify(user_input)

    if confidence < 0.5:
        return "I'm not sure what you meant — could you rephrase that?"

    if intent in ["Swearing", "SexualQuery"]:
        return "Let's keep our conversation respectful, please."

    if intent in ["Greeting", "Thanks", "GoodBye"]:
        return f"{intent_clf.intents[intent][0]} (detected intent: {intent})"

    # Forward to empathy or assessment pipeline
    return None