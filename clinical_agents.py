"""
clinical_agents.py
────────────────────────────
ClinicalAgentOrchestrator — Multi-agent coordinator for Zenark.
Coordinates:
 • EmotionAgent
 • EmpathyAgent
 • NavigationAgent
 • QuestionAgent
 • JudgmentAgent
 • WellnessAgent
"""

import logging
import asyncio
import os
from typing import Dict, Any, List, Optional, cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from services.zenark_db_cloud import (
    get_user,
    get_conversation_history,
    save_conversation,
    save_interaction,
)

from agents.emotion_agent import EmotionAgent
from agents.empathy_agent import EmpathyAgent
from agents.navigation_agent import NavigationAgent
from agents.question_agent import QuestionAgent
from agents.judgment_agent import JudgmentAgent
from agents.wellness_agent import WellnessAgent


logger = logging.getLogger("zenark.agent.orchestrator")


# =====================================================
# 🔧 Async-safe wrapper for Streamlit / Motor operations
# =====================================================
def run_async_task(coro):
    """Runs a coroutine safely across Streamlit event loops."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return asyncio.ensure_future(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)


# =====================================================
# 🧠 Clinical Agent Orchestrator
# =====================================================
class ClinicalAgentOrchestrator:
    MAX_QUESTIONS = 30  # stop before or at 30th question
    CONFIDENCE_STOP_THRESHOLD = 0.85  # early stop if confidence > 0.85

    def __init__(self):
        self.agent_name = "ClinicalAgentOrchestrator"
        self.emotion_agent = EmotionAgent()
        self.empathy_agent = EmpathyAgent()
        self.navigation_agent = NavigationAgent()
        self.question_agent = QuestionAgent()
        self.judgment_agent = JudgmentAgent()
        self.wellness_agent = WellnessAgent()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.question_counter = 0
        self.category_scores: Dict[str, float] = {}
        logger.info("🧩 ClinicalAgentOrchestrator initialized successfully.")

    # =====================================================
    # 💬 Core orchestration pipeline
    # =====================================================
    async def process_user_message(
        self,
        username: str,
        session_id: str,
        user_message: str,
        user_context: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Full orchestration for one user input → Zenark pipeline."""

        try:
            history = conversation_history
            logger.info(
                f"📂 Loaded context for user={username}, previous messages={len(history)}"
            )

            # 🧮 Stop condition check
            if self.question_counter >= self.MAX_QUESTIONS:
                logger.info("🛑 Reached maximum question limit — generating final summary")
                return await self._generate_final_assessment(username, session_id, history)

            # 2️⃣ Emotion analysis
            raw_emotion = self.emotion_agent.run(user_message, username)
            emotion_summary = raw_emotion if isinstance(raw_emotion, dict) else {}
            if not emotion_summary:
                logger.warning("⚠️ Empty emotion_summary recovered; using fallback")

            # 3️⃣ Empathy agent (async)
            empathy_text, emotion_summary_updated = await self.empathy_agent.run(
                user_message, username, session_id
            )
            emotion_summary.update(emotion_summary_updated)
            logger.info("💬 EmpathyAgent completed successfully")

            # 4️⃣ Navigation: choose next adaptive question
            nav_context = {
                "phase": "adaptive",
                "category_scores": emotion_summary.get("clinical_signals", {}),
                "target_category": emotion_summary.get("dominant_emotion", "general"),
                "questions_asked": self.question_counter,
            }

            result_q = self.navigation_agent.run(nav_context, user_message)
            next_q = cast(Dict[str, Any], result_q or {})

            # 🧭 NEW: Detect navigation conclusion → trigger WellnessAgent
            if next_q and next_q.get("intent") == "conclusion":
                final_category = next_q.get("category", "general")
                reason = next_q.get("reason", "emotional distress pattern")

                logger.info(
                    f"[Orchestrator] 🌿 Navigation complete — transitioning to WellnessAgent ({final_category})"
                )

                # Prepare a minimal judgment context for WellnessAgent
                judgment_summary = {
                    "summary": {
                        "primary_category": final_category,
                        "confidence": 0.85,
                    }
                }

                wellness_message, wellness_meta = self.wellness_agent.run(
                    emotion_profile=emotion_summary,
                    judgment_summary=judgment_summary,
                    user_input="yes",  # Automatically opt-in to wellness suggestion
                    user_id=username,
                    session_id=session_id,
                )

                await save_interaction(
                    user_id=username,
                    agent="WellnessAgent",
                    user_input="[SYSTEM_TRIGGER]",
                    response=wellness_message,
                    metadata={
                        "final_category": final_category,
                        "reason": reason,
                        "phase": "wellness",
                    },
                )

                return {
                    "phase": "wellness",
                    "message": wellness_message,
                    "meta": wellness_meta,
                    "category": final_category,
                    "reason": reason,
                }

            # 5️⃣ QuestionAgent — combine empathy + adaptive question
            merged_prompt, q_metadata = await self.question_agent.run(
                username, next_q, empathy_text, session_id
            )

            # 6️⃣ Judgment analysis
            evaluated_data, wellness_trigger_msg = self.judgment_agent.run(
                emotion_summary, username
            )

            # 🧮 Track category scores
            for cat, score in evaluated_data.get("scores", {}).items():
                self.category_scores[cat] = self.category_scores.get(cat, 0) + score

            # 🧩 Early stop if confident enough
            if self.category_scores:
                top_category = max(self.category_scores, key=lambda k: self.category_scores[k])
                confidence = (
                    self.category_scores[top_category]
                    / sum(self.category_scores.values())
                )
                if confidence >= self.CONFIDENCE_STOP_THRESHOLD:
                    logger.info(
                        f"🛑 Early stop: high confidence ({confidence:.2f}) in {top_category}"
                    )
                    return await self._generate_final_assessment(
                        username, session_id, history
                    )

            # 7️⃣ Optional wellness agent (post-response suggestion)
            wellness_suggestion = None
            wellness_method = getattr(self.wellness_agent, "run", None)
            if callable(wellness_method):
                if asyncio.iscoroutinefunction(wellness_method):
                    wellness_suggestion = await wellness_method(
                        emotion_summary,
                        evaluated_data,
                        user_message,
                        username,
                        session_id,
                    )
                else:
                    wellness_suggestion = wellness_method(
                        emotion_summary,
                        evaluated_data,
                        user_message,
                        username,
                        session_id,
                    )

            # ✅ Save interactions
            emotion_scores = emotion_summary.get("emotion_scores")
            await save_conversation(username, session_id, "User", user_message, emotion_scores)
            await save_conversation(username, session_id, "Zen", merged_prompt, emotion_scores)
            await save_interaction(
                user_id=username,
                agent=self.agent_name,
                user_input=user_message,
                response=merged_prompt,
                metadata={
                    "emotion_summary": emotion_summary,
                    "judgment": evaluated_data,
                    "wellness_trigger": wellness_trigger_msg,
                    "question_metadata": q_metadata,
                },
            )

            logger.info("💾 Conversation state successfully persisted to MongoDB")

            return {
                "empathetic_response": empathy_text,
                "diagnostic_question": next_q,
                "final_prompt": merged_prompt,
                "emotion_summary": emotion_summary,
                "judgment_summary": evaluated_data,
                "wellness_suggestion": wellness_suggestion,
                "remaining_questions": self.MAX_QUESTIONS - self.question_counter,
            }

        except Exception as e:
            logger.exception("❌ Orchestrator error")
            return {
                "error": str(e),
                "fallback_response": (
                    "I'm here with you. Let's take this one step at a time — "
                    "tell me more about how you're feeling right now."
                ),
            }


    # =====================================================
    # 🧠 Final assessment summary
    # =====================================================
    async def _generate_final_assessment(
        self,
        username: str,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Called when max question count or early-stop threshold reached."""
        if not self.category_scores:
            return {
                "final_summary": (
                    "We've explored your thoughts and feelings for a while. "
                    "Although no specific category clearly stands out, this conversation "
                    "shows resilience and a willingness to reflect deeply. 💬"
                ),
                "category": "undetermined",
                "confidence": 0.0,
            }

        # Determine top category
        top_category = max(self.category_scores.items(), key=lambda x: x[1])[0]
        confidence = self.category_scores[top_category] / sum(self.category_scores.values())

        # Generate reasoning summary using GPT-4o-mini
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "Summarize the conversation and infer the psychological category "
                    "based on emotional cues."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Category: {top_category}\n\nConversation:\n"
                    + "\n".join(
                        [
                            f"{m.get('role', 'user')}: {m.get('content', str(m))}"
                            for m in history[-20:]
                        ]
                    )
                ),
            },
        ]

        reasoning = ""
        try:
            result = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            content = getattr(result.choices[0].message, "content", None)
            reasoning = content.strip() if isinstance(content, str) else "No reasoning returned."
        except Exception as e:
            reasoning = f"Summary generation failed: {e}"

        final_msg = (
            f"🧠 **Preliminary Insight:** Based on our discussion, your responses "
            f"most closely align with **{top_category.title()}** indicators.\n\n"
            f"**Confidence:** {confidence:.2f}\n\n"
            f"**Reasoning:** {reasoning}\n\n"
            "This isn't a diagnosis — it’s an interpretation to guide reflection. "
            "Would you like me to suggest next steps or coping strategies?"
        )

        await save_interaction(
            user_id=username,
            agent=self.agent_name,
            user_input="[SYSTEM_TRIGGER]",
            response=final_msg,
            metadata={
                "final_category": top_category,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        )

        return {
            "final_summary": final_msg,
            "category": top_category,
            "confidence": confidence,
        }
