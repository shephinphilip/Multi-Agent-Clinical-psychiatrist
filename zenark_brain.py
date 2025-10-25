"""
zenark_brain.py
────────────────────────────
Central Cognitive Orchestrator for Zenark
────────────────────────────
Handles:
 • Smart conversation initiation (via ConversationInitiator)
 • Multi-agent emotional assessment workflow (via ClinicalAgentOrchestrator)

This acts as the brain of Zenark’s psychological dialogue system.
All async DB operations must be handled externally and passed in.
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from services.zenark_db_cloud import get_user, get_conversation_history

# ──────────────────────────────
# Import Agent Systems
# ──────────────────────────────
from agents.conversation_initiator import ConversationInitiator
from clinical_agents import ClinicalAgentOrchestrator

logger = logging.getLogger("zenark.brain")


class ZenarkBrain:
    """
    Core orchestrator — NO DIRECT DATABASE ACCESS.
    All MongoDB reads/writes must be done outside and passed in.
    """

    def __init__(self):
        self.agent_name = "ZenarkBrain"
        self.initiator = ConversationInitiator()
        self.clinical = ClinicalAgentOrchestrator()
        logger.info("🧠 ZenarkBrain initialized successfully.")

    # ──────────────────────────────
    async def start_conversation(
        self,
        username: str,
        session_id: str,
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Start a new conversation.
        If user_context is provided, use it. Otherwise, treat as fresh.
        Database creation happens externally.
        """
        try:
            logger.info(f"👋 Initiating conversation for '{username}'")
            conversation_history = conversation_history or []

            if user_context:
             _ = user_context.get("username", "")


            # Generate greeting
            greeting_text = await self.initiator.initiate_chat(username, session_id)

            logger.info(f"✅ Greeting generated for user={username}")
            return {
                "phase": "welcome",
                "greeting": greeting_text,
                "session_id": session_id,
            }

        except Exception as e:
            logger.exception("❌ Error starting conversation")
            return {
                "error": str(e),
                "fallback": f"Hey {username.title()}, I’m Zen — how are you feeling today?",
            }

    # ──────────────────────────────
    async def continue_conversation(
        self,
        username: str,
        session_id: str,
        user_message: str,
        user_context: Dict[str, Any],
        conversation_history: list,
    ) -> Dict[str, Any]:
        """
        Continue adaptive clinical dialogue.
        All DB data is pre-loaded and passed in.
        """
        try:
            logger.info(f"[ZenarkBrain] Continuing conversation for {username}")

            # Pass all context into agent — no DB calls inside!
            response = await self.clinical.process_user_message(
                username=username,
                session_id=session_id,
                user_message=user_message,
                user_context=user_context,
                conversation_history=conversation_history,
            )

            return {
                "phase": "adaptive",
                "output": response,
            }

        except Exception as e:
            logger.exception("❌ Error continuing conversation")
            return {
                "error": str(e),
                "fallback_response": "Let’s pause. How are you feeling right now?",
            }
        

    # ------------------------------------------------------------
    # ✅ NEW: Context Fetcher for Streamlit UI
    # ------------------------------------------------------------
    async def get_context(self, username: str, session_id: str):
        """
        Fetch the user's stored profile and conversation history safely.
        Returns: (user_doc, conversation_history)
        """
        try:
            user_doc = await get_user(username)
            history = await get_conversation_history(username, session_id)
            logger.info(f"📚 Context fetched for {username}: {len(history)} past messages")
            return user_doc or {}, history or []
        except Exception as e:
            logger.error(f"⚠️ Error fetching context for {username}: {e}")
            return {}, []