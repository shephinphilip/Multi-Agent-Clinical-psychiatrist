"""
agents/empathy_agent.py
────────────────────────────
Empathy Agent for Zenark's Multi-Agent System

Dynamic version — no hardcoded templates.
Uses EmotionAgent for emotional signal detection,
then generates contextual empathy through LLM reasoning.
"""

import logging
import asyncio
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from agents.emotion_agent import EmotionAgent
from services.zenark_db_cloud import save_interaction, get_conversation_history
from openai import OpenAI
import os

logger = logging.getLogger("zenark.agent.empathy")

# ──────────────────────────────
# Async-safe runner
# ──────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ──────────────────────────────
# Empathy Agent (Dynamic)
# ──────────────────────────────
class EmpathyAgent:
    """Dynamic empathy generator using detected emotion + conversation context."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.emotion_agent = EmotionAgent()
        self.agent_name = "EmpathyAgent"
        logger.info("🤗 EmpathyAgent initialized — LLM-driven empathy enabled.")

    # ──────────────────────────────
    # Core Function
    # ──────────────────────────────
    async def run(self, user_input: str, username: str, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Detect emotion, generate empathetic response via LLM, and save interaction.
        Returns: (empathetic_message, full_emotion_summary)
        """
        if not user_input or not isinstance(user_input, str):
            logger.warning("⚠️ EmpathyAgent received empty input.")
            return "I’m here whenever you’d like to talk.", {"dominant_emotion": "neutral"}

        # Step 1: Detect emotion using EmotionAgent
        emotion_summary = self.emotion_agent.run(user_input, username)
        dominant = emotion_summary.get("dominant_emotion", "neutral")

        # Step 2: Retrieve recent conversation for context
        conversation_history = []
        try:
            if session_id:
                conversation_history = await get_conversation_history(username, session_id, limit=5)
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch conversation history: {e}")

        recent_text = "\n".join(
            [f"{c['speaker']}: {c['message']}" for c in conversation_history[-5:]]
        ) if conversation_history else "First interaction with this user."

        # Step 3: Build LLM prompt for empathy generation
        system_prompt = f"""
        You are Zen, an empathetic conversational AI companion.
        Your goal is to respond warmly and naturally, showing understanding of the user's feelings.

        Rules:
        - NEVER use clichés like "It sounds like" or "I know how you feel."
        - Reference the user’s emotion when appropriate ({dominant})
        - Keep the tone human, warm, and context-aware.
        - Write 2–4 sentences max.
        - Use continuity from previous messages if possible.
        - Avoid questions unless it's a gentle invitation to share more.
        - No repetitive or robotic phrases.
        """

        user_prompt = f"""
        Recent conversation:
        {recent_text}

        The user just said: "{user_input}"
        Detected emotion: {dominant.upper()}
        Generate one empathetic response that naturally fits this emotional state.
        """

        # Step 4: Generate empathetic reflection
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.85,
                max_tokens=120
            )

            content = getattr(response.choices[0].message, "content", None)
            if content is not None:
                empathetic_reply = content.strip()
            else:
                empathetic_reply = (
                    "I can tell this means a lot to you. It’s okay to feel this way — I’m here to listen and help you process it at your own pace."
                )

        except Exception as e:
            logger.error(f"❌ LLM empathy generation failed: {e}")
            empathetic_reply = (
                "I can sense this means a lot to you. You don’t have to hold it all in — I’m here with you, and it’s okay to feel this way."
            )

        # Step 5: Prepare metadata
        metadata = {
            "dominant_emotion": dominant,
            "emotion_scores": emotion_summary.get("emotion_scores", {}),
            "clinical_signals": emotion_summary.get("clinical_signals", {}),
            "intensity": emotion_summary.get("intensity", 0.0),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Step 6: Save empathetic interaction asynchronously
        try:
            run_async(
                save_interaction(
                    user_id=username,
                    agent=self.agent_name,
                    user_input=user_input,
                    response=empathetic_reply,
                    metadata=metadata,
                )
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save EmpathyAgent interaction: {e}")

        logger.info(f"[EmpathyAgent] ❤️ Emotion={dominant} | Reply='{empathetic_reply}'")
        return empathetic_reply, emotion_summary
