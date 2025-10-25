"""
agents/wellness_agent.py
────────────────────────────
LLM-driven Wellness / Support Agent grounded in real
"Complete Breathing Guide for Mental Well-being."

This version lets GPT use the curated breathing guide content
to produce empathetic, evidence-based self-help suggestions
based on the user's emotional and diagnostic profile.
"""

import logging
import asyncio
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from services.zenark_db_cloud import save_interaction, get_conversation_history

logger = logging.getLogger("zenark.agent.wellness")
client = OpenAI()

# ──────────────────────────────
# Async-safe helper
# ──────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ──────────────────────────────
# WellnessAgent
# ──────────────────────────────
class WellnessAgent:
    """Context-aware, LLM-driven, clinically-grounded Support Agent."""

    def __init__(self, model: str = "gpt-4o-mini", history_limit: int = 8):
        self.agent_name = "WellnessAgent"
        self.model = model
        self.history_limit = history_limit
        logger.info("🌿 WellnessAgent initialized with breathing-guide grounding.")

        # Short, embedded summary of your real guide
        self.breathing_reference = BREATHING_GUIDE_SUMMARY

    # ──────────────────────────────
    # MAIN ENTRY
    # ──────────────────────────────
    def run(
        self,
        emotion_profile: Dict[str, Any],
        judgment_summary: Dict[str, Any],
        user_input: Optional[str],
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle user consent + generate LLM-based suggestion."""
        user_text = (user_input or "").strip().lower()

        if not judgment_summary:
            return (
                "I'm here if you'd like a small moment to relax. Would you like me to suggest something calming?",
                {"phase": "init"},
            )

        # Ask permission
        if user_text in ["", "maybe", "not sure"]:
            return (
                "Would you like a short, personalized wellness activity — like a breathing or grounding exercise — to feel better?",
                {"phase": "offer"},
            )

        if user_text in ["no", "nah", "not now"]:
            return (
                "That’s totally okay. Resting is also self-care. I’ll be here when you’re ready.",
                {"phase": "end"},
            )

        if user_text in ["yes", "okay", "sure", "please"]:
            suggestion = self._generate_grounded_recommendation(
                user_id, session_id, emotion_profile, judgment_summary
            )
            run_async(
                save_interaction(
                    user_id=user_id,
                    user_input="(accepted wellness suggestion)",
                    agent=self.agent_name,
                    response=suggestion,
                    metadata={
                        "emotion_profile": emotion_profile,
                        "judgment_summary": judgment_summary,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            )
            return suggestion, {"phase": "recommendation"}

        return (
            "Would you like me to guide you through a quick relaxation or breathing technique?",
            {"phase": "clarify"},
        )

    # ──────────────────────────────
    # LLM GENERATOR (grounded)
    # ──────────────────────────────
    def _generate_grounded_recommendation(
        self,
        user_id: str,
        session_id: Optional[str],
        emotion_profile: Dict[str, Any],
        judgment_summary: Dict[str, Any],
    ) -> str:
        """Generate suggestion using context + breathing guide grounding."""
        # Load last few messages
        try:
            history = run_async(get_conversation_history(user_id, session_id, limit=self.history_limit))
        except Exception as e:
            logger.warning(f"⚠️ Could not load chat history: {e}")
            history = []

        recent_chat = "\n".join(f"{h['speaker']}: {h['message']}" for h in history) if history else "No previous chat."

        dominant = emotion_profile.get("dominant_emotion", "neutral")
        category = judgment_summary.get("summary", {}).get("primary_category", "general_wellness")
        confidence = judgment_summary.get("summary", {}).get("confidence", 0.6)

        prompt = f"""
You are a calm, licensed AI wellness assistant helping a user after an emotional-diagnostic chat.
Use the below **scientifically backed Breathing Guide** to select the most relevant technique
and generate a short, compassionate suggestion.

User Context:
- Dominant Emotion: {dominant}
- Mental Health Category: {category}
- Confidence: {confidence}
- Recent Conversation:
{recent_chat}

Breathing Guide Reference (extract):
{self.breathing_reference}

Instructions:
• Select ONE breathing or mindfulness technique that fits the user's category.
• Write in 2–3 warm, supportive sentences, using natural tone (no lists or jargon).
• Mention the technique name briefly and describe “how” to do it, not as rigid steps.
• Avoid medical claims, and end with gentle reassurance.
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an empathetic, evidence-based AI therapist assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,
                temperature=0.7,
            )
            choice = response.choices[0] if response.choices else None
            message = getattr(choice, "message", None)
            content = getattr(message, "content", "")
            suggestion = (content or "").strip() or "Let's take a deep breath together."

            logger.info(f"[WellnessAgent] 🌬️ Suggestion: {suggestion}")
            return suggestion

        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return (
                "Try a slow 4-7-8 breath — inhale for 4 seconds, hold for 7, exhale for 8. "
                "It can calm your body and ease tension right now."
            )
    

# ──────────────────────────────
# BREATHING GUIDE SNIPPET (summarized for grounding)
# ──────────────────────────────
BREATHING_GUIDE_SUMMARY = """
Anxiety → Box Breathing (4-4-4-4) or 4-7-8 Breathing to slow the heart rate.
Panic → Grounding Breath (5-2-5) or Humming Breath to regain control.
Nervousness → Equal Breathing or Affirmation Breathing (“I am calm”).
Anger → Cooling Breath or Lion’s Breath to release tension.
Restlessness → Coherent Breathing (5-5) to balance rhythm.
Insomnia → 4-7-8 Breathing or Body-Scan + Deep Breathing for sleep.
Depression → Energizing Breath or Morning Uplift Breath to raise energy.
All are gentle, safe, non-clinical methods proven to relax the nervous system.
"""


