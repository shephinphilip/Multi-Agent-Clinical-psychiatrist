"""
agents/question_agent.py
────────────────────────────
Question Agent (LLM-driven)
────────────────────────────
Fetches conversation context & user memory directly from the Zenark cloud DB,
then crafts an emotionally intelligent, adaptive question.
"""

import logging
from typing import Dict, Any, Tuple, List
import asyncio
from openai import AsyncOpenAI

from services.zenark_db_cloud import (
    get_conversation_history,
    get_user,
    save_interaction,
)

logger = logging.getLogger("zenark.agent.question")
client = AsyncOpenAI()

def run_async(coro):
    """Ensure async functions can safely run from sync contexts."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)


class QuestionAgent:
    """LLM-based empathetic question generator (with cloud context)."""

    def __init__(self):
        self.agent_name = "QuestionAgent"
        logger.info("💬 QuestionAgent (LLM) initialized successfully.")

    async def run(
        self,
        user_id: str,
        nav_question: Dict[str, Any] | None,
        empathy_reply: str | None,
        session_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an empathetic, adaptive question using:
        - Conversation history (from MongoDB)
        - User memory (from MongoDB)
        - Navigation question + empathy tone
        """
        if not nav_question:
            logger.warning("⚠️ No navigation question provided.")
            return ("I think we’ve covered a lot today. Would you like to pause here?", {})

        # ──────────────────────────────
        # Cloud context fetch
        # ──────────────────────────────
        conv_docs = await get_conversation_history(user_id, session_id=session_id, limit=3)
        user_doc = await get_user(user_id) or {}

        conversation_history = [
            f"{msg.get('speaker', 'user')}: {msg.get('message', '')}" for msg in conv_docs
        ]

        memory_context = [
            f"User role: {user_doc.get('role', 'unknown')}",
            f"Age Band: {user_doc.get('age_band', 'not specified')}",
            f"Language: {user_doc.get('language', 'en')}",
        ]

        question_text = (nav_question.get("text") or "").strip()
        category = nav_question.get("category") or "general"
        intent = nav_question.get("intent") or "diagnostic"
        empathy_tone = (empathy_reply or "").strip()

        # ──────────────────────────────
        # Build LLM prompt
        # ──────────────────────────────
        system_prompt = f"""
You are Zen, a warm and empathetic AI companion helping users explore their emotions safely.
Rephrase this clinical question empathetically for a 13-year-old adolescent.Keep its meaning the same, but make it sound supportive and natural.

STRICT RULES:
- Never use phrases like "It sounds like" or "You're going through a lot".
- Avoid clinical or robotic tones.
- Be conversational, gentle, and curious.
- Connect your next question to what the user said recently.
- Only ask one question at a time.
- Make it sound like a natural flow, not an interview.

CONVERSATION HISTORY:
{conversation_history[-3:] if conversation_history else 'Starting conversation'}

USER PROFILE:
{memory_context}

CURRENT QUESTION CONTEXT:
Category: {category}
Intent: {intent}
Empathy Reply: {empathy_tone}

Your goal:
Blend empathy with the navigation question below into a natural follow-up prompt.
It should feel like a genuine human conversation, not a form.

Base Question:
"{question_text}"
"""

        # ──────────────────────────────
        # Query LLM
        # ──────────────────────────────
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the next message in Zen’s voice."},
                ],
                temperature=0.8,
            )
            final_prompt = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            final_prompt = (
                f"{empathy_tone} {question_text}".strip()
                or "Let's continue. How have you been feeling lately?"
            )

        # ──────────────────────────────
        # Save to DB
        # ──────────────────────────────
        metadata = {
            "category": category,
            "intent": intent,
            "nav_question": question_text,
            "empathy_used": empathy_tone,
            "conversation_context": conversation_history,
            "memory_context": memory_context,
            "final_prompt": final_prompt,
        }

        try:
            await save_interaction(
                user_id=user_id,
                user_input=question_text,
                response=final_prompt,
                metadata=metadata,
                agent=self.agent_name,
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save QuestionAgent output: {e}")

        logger.info(f"[QuestionAgent] 🧠 Generated empathetic question: {final_prompt}")
        return final_prompt, metadata
