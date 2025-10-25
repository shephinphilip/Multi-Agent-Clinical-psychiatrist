"""
agents/emotion_agent.py
────────────────────────────
Emotion Agent for Zenark's Multi-Agent System

Uses NRC Hashtag Emotion Lexicon (v0.2) for deep lexical emotion detection.
Maps detected emotions to clinical mental-health categories (e.g., depression, anxiety).
Integrates with the system memory manager and other agents.
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime
from utils.nrc_emotion_analyzer import NRCEmotionAnalyzer
from services.zenark_db_cloud import save_interaction  # async function

logger = logging.getLogger("zenark.agent.emotion")


# Utility: safe runner for async calls inside sync methods
def run_async(coro):
    """Ensure async functions run even in sync context."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)


class EmotionAgent:
    """Emotion Detection Agent — wraps NRCEmotionAnalyzer for agentic usage."""

    def __init__(self, lexicon_path: str = "dataset/NRC-Hashtag-Emotion-Lexicon-v0.2.txt"):
        self.analyzer = NRCEmotionAnalyzer(lexicon_path)
        self.agent_name = "EmotionAgent"
        logger.info(f"🤖 EmotionAgent initialized with lexicon: {lexicon_path}")

    def run(self, user_input: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Main entry point for the EmotionAgent.
        Detects emotional tone, intensity, and clinical signals from user text.
        """
        if not user_input or not isinstance(user_input, str):
            logger.warning("⚠️ EmotionAgent received empty or invalid input.")
            return {"dominant_emotion": "neutral", "emotion_scores": {"neutral": 1.0}}

        logger.info(f"[EmotionAgent] 🧠 Analyzing user input: {user_input}")

        # Step 1. Perform NRC-based emotion analysis
        summary = self.analyzer.get_emotion_summary(user_input)

        dominant = summary.get("dominant_emotions", [])
        top_emotion = dominant[0][0] if dominant else "neutral"
        emotion_scores = summary.get("all_emotions", {})
        clinical_signals = summary.get("clinical_signals", {})
        intensity = summary.get("intensity", 0.0)
        charged = summary.get("is_charged", False)

        # Step 2. Prepare metadata for DB logging
        metadata = {
            "dominant_emotion": top_emotion,
            "emotion_scores": emotion_scores,
            "clinical_signals": clinical_signals,
            "intensity": intensity,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Step 3. Safely log to MongoDB (async wrapper)
        try:
            run_async(
                save_interaction(
                    user_id=user_id,
                    user_input=user_input,
                    agent="EmotionAgent",
                    response=top_emotion,
                    metadata=metadata
                )
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save interaction: {e}")

        logger.info(f"[EmotionAgent] 🎭 Dominant: {top_emotion} | Intensity: {intensity:.3f}")

        # Step 4. Return structured result
        return {
            "dominant_emotion": top_emotion,
            "emotion_scores": emotion_scores,
            "clinical_signals": clinical_signals,
            "intensity": intensity,
            "is_charged": charged,
            "matched_words": summary.get("matched_words", [])
        }
