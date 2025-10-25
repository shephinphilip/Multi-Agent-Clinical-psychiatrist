"""
agents/judgment_agent.py
────────────────────────────
Judgment Agent for Zenark — MAGI-aligned evaluation layer.

Evaluates emotion and clinical signals from EmotionAgent, determines
if the user's state satisfies a diagnostic criterion, and decides
whether to proceed to the next question node or request clarification.

Schema Output → {
    "criterion_met": bool,
    "confidence": float,
    "next_node": str,
    "evidence": list[str],
    "summary": {...}
}

Flow:
  EmotionAgent → JudgmentAgent → NavigationAgent → WellnessAgent
"""

import logging
import asyncio
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from services.zenark_db_cloud import save_interaction  # async DB call

logger = logging.getLogger("zenark.agent.judgment")


# Utility: async-safe runner
def run_async(coro):
    """Run async function safely within sync context."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        return asyncio.run(coro)


class JudgmentAgent:
    """Evaluates emotion & clinical signals to determine diagnostic transitions."""

    def __init__(self):
        self.agent_name = "JudgmentAgent"
        # Three decision thresholds (per MAGI paper)
        self.thresholds = {
            "direct_match": 0.8,
            "semantic_match": 0.6,
            "ambiguous": 0.4,
        }
        logger.info("⚖️ JudgmentAgent initialized with MAGI evaluation schema.")

    # ──────────────────────────────
    # CORE LOGIC
    # ──────────────────────────────
    def evaluate_signals(self, emotion_summary: Dict[str, Any], current_node: str = "A0") -> Dict[str, Any]:
        """
        Evaluate EmotionAgent output to determine if a diagnostic criterion is met.

        Args:
            emotion_summary: Output from EmotionAgent.run()
            current_node: Current diagnostic node ID (e.g., 'A3a')

        Returns:
            Structured evaluation dict.
        """
        if not emotion_summary or "clinical_signals" not in emotion_summary:
            logger.warning("⚠️ No emotion signals provided to JudgmentAgent.")
            return {
                "criterion_met": False,
                "confidence": 0.0,
                "next_node": current_node,
                "evidence": [],
                "summary": {"criterion_met": False, "confidence": 0.0, "next_node": current_node},
            }

        clinical = emotion_summary.get("clinical_signals", {})
        evaluated_map: Dict[str, Any] = {}
        primary_category, primary_score = None, 0.0

        # Determine highest scoring category (dominant signal)
        for category, score in clinical.items():
            evaluated_map[category] = round(score, 3)
            if score > primary_score:
                primary_category, primary_score = category, score

        # Compute confidence based on score severity
        confidence = self._compute_confidence(primary_score)

        # Criterion validation based on threshold
        criterion_met = confidence >= self.thresholds["semantic_match"]
        next_node = self._determine_next_node(current_node, criterion_met)

        evidence = self._extract_evidence(emotion_summary, primary_category)

        result = {
            "criterion_met": criterion_met,
            "confidence": round(confidence, 3),
            "next_node": next_node,
            "evidence": evidence,
            "summary": {
                "criterion_met": criterion_met,
                "confidence": round(confidence, 3),
                "next_node": next_node,
                "primary_category": primary_category,
                "primary_score": round(primary_score, 3),
            },
        }

        logger.info(
            f"[JudgmentAgent] Node={current_node} | "
            f"Category={primary_category} | "
            f"Score={primary_score:.2f} | "
            f"CriterionMet={criterion_met} | "
            f"Confidence={confidence:.2f}"
        )
        return result

    # ──────────────────────────────
    # SUPPORT FUNCTIONS
    # ──────────────────────────────

    def _determine_next_node(self, current_node: str, criterion_met: bool) -> str:
        """Simple deterministic transition (e.g., A3a → A3b if criterion met)."""
        if not current_node:
            return "A0"
        if criterion_met:
            # e.g., A3a → A3b
            if current_node[-1].isalpha():
                return current_node[:-1] + chr(ord(current_node[-1]) + 1)
            else:
                return f"{current_node}a"
        else:
            return current_node  # stay until clarification

    def _extract_evidence(self, emotion_summary: Dict[str, Any], primary: Optional[str]) -> List[str]:
        """Extract key emotional indicators for interpretability."""
        matched = emotion_summary.get("matched_words", [])
        dominant = emotion_summary.get("dominant_emotion", "")
        return list(set([dominant] + matched)) if primary else matched
    
    # ──────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────
    def _compute_confidence(self, score: float) -> float:
        if score >= 0.8:
            return 0.9
        elif score >= 0.6:
            return 0.7
        elif score >= 0.4:
            return 0.5
        elif score > 0:
            return 0.3
        return 0.0

    def _next_category(self, current_category: str, criterion_met: bool) -> str:
        """Move to next category only if criterion_met=True."""
        categories = ["depression", "anxiety", "ptsd", "ocd", "bipolar", "general_stress"]
        if current_category not in categories:
            return "general_stress"
        idx = categories.index(current_category)
        return categories[idx + 1] if criterion_met and idx + 1 < len(categories) else current_category

    def _empty_result(self, category: str) -> Dict[str, Any]:
        return {
            "criterion_met": False,
            "confidence": 0.0,
            "next_category": category,
            "evidence": [],
            "summary": {"criterion_met": False, "confidence": 0.0, "next_category": category},
    }

    # ──────────────────────────────
    # INTEGRATION PIPELINE
    # ──────────────────────────────
    def run(self, emotion_summary: Dict[str, Any], user_id: str = "anonymous", current_node: str = "A0") -> Tuple[Dict[str, Any], str]:
        """
        Public entry point for integration with NavigationAgent.

        Args:
            emotion_summary: dict → EmotionAgent.run() output
            user_id: str → current session user ID
            current_node: str → current diagnostic node ID

        Returns:
            Tuple → (evaluation_result, feedback_message)
        """
        if not emotion_summary:
            logger.warning("⚠️ Empty input to JudgmentAgent.")
            return {}, "No emotional data available for evaluation."

        evaluated = self.evaluate_signals(emotion_summary, current_node)
        summary = evaluated.get("summary", {})
        criterion_met = summary.get("criterion_met", False)
        confidence = summary.get("confidence", 0.0)
        primary_category = summary.get("primary_category", "general")

        # Human-readable feedback
        feedback = (
            f"Criterion satisfied with {confidence:.2f} confidence. Proceed to {summary.get('next_node')}."
            if criterion_met
            else f"Criterion not satisfied ({confidence:.2f}). Awaiting clarification before moving forward."
        )

        # Asynchronous DB logging
        metadata = {
            "evaluation": evaluated,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            run_async(
                save_interaction(
                    user_id=user_id,
                    user_input="(system evaluation)",
                    agent=self.agent_name,
                    response=feedback,
                    metadata=metadata,
                )
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save JudgmentAgent interaction: {e}")

        return evaluated, feedback
