"""
agents/navigation_agent.py — v3.1 (QuestionPool-driven + Wellness-ready)
────────────────────────────────────────────
Emotion-aware, question-pool-driven navigation system for Zenark.
Automatically concludes after 30 questions and transitions to WellnessAgent.
"""

import logging, random
from typing import Dict, Any, Optional, List
from utils.data import MentalHealthDataset
from agents.judgment_agent import JudgmentAgent
from agents.wellness_agent import WellnessAgent

logger = logging.getLogger("zenark.agent.navigation")


class NavigationAgent:
    """Adaptive conversational navigator reading directly from question_pool.json."""

    def __init__(self, dataset_path: str = "question_pool"):
        self.dataset = MentalHealthDataset(dataset_path)
        self.questions = self.dataset.questions_by_category  # e.g. {"family": [...], "friends": [...]}
        self.judger = JudgmentAgent()
        self.wellness = WellnessAgent()
        self.domain_depth = {}
        self.max_depth_per_domain = 3
        self.total_questions_asked = 0
        self.max_questions = 30
        self.category_scores = {}  # track which category dominates

        logger.info(
            f"🧭 NavigationAgent v3.1 initialized with "
            f"{sum(len(v) for v in self.questions.values())} questions."
        )

    # ──────────────────────────────
    def run(self, context: Dict[str, Any], user_response: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        - If no response: start conversation with an opening question.
        - If response given: judge, update depth, and decide next question or conclude.
        """

        # 1️⃣ Start new conversation
        if user_response is None:
            logger.info("[NavigationAgent] 🟢 Starting new conversation flow.")
            self.total_questions_asked = 0
            self.category_scores.clear()
            self.domain_depth.clear()
            return self._select_next(context)

        # 2️⃣ Evaluate response confidence
        logger.info("[NavigationAgent] 🧩 Evaluating last response via JudgmentAgent...")
        eval_result, _ = self.judger.run(context.get("emotion_summary", {}))
        confidence = eval_result.get("summary", {}).get("confidence", 0.0)
        criterion_met = eval_result.get("summary", {}).get("criterion_met", True)
        dominant_emotion = context.get("emotion_summary", {}).get("dominant", "")

        # 3️⃣ Detect domain category
        category = self._detect_category(context, user_response)
        self.domain_depth[category] = self.domain_depth.get(category, 0) + 1
        depth = self.domain_depth[category]

        # 4️⃣ Track totals
        self.total_questions_asked += 1
        self.category_scores[category] = self.category_scores.get(category, 0) + 1

        logger.info(
            f"[NavigationAgent] Domain={category}, Depth={depth}, Emotion={dominant_emotion}, "
            f"Confidence={confidence:.2f}, TotalQ={self.total_questions_asked}"
        )

        # 5️⃣ Stop condition — reached 30 questions
        if self.total_questions_asked >= self.max_questions:
            final_category, reason = self._derive_final_category_and_reason()
            logger.info(
                f"[NavigationAgent] 🏁 Reached {self.max_questions} questions — concluding with {final_category}."
            )
            return {
                "intent": "conclusion",
                "category": final_category,
                "reason": reason,
                "text": self._generate_final_summary(final_category, reason),
            }

        # 6️⃣ Low confidence → reflective question
        if confidence < 0.6:
            return self._make_reflective_followup(category, dominant_emotion)

        # 7️⃣ Continue exploring same domain
        if depth < self.max_depth_per_domain:
            return self._make_contextual_followup(category)

        # 8️⃣ Move to next unexplored domain or conclude
        if criterion_met and depth >= self.max_depth_per_domain:
            logger.info("[NavigationAgent] ✅ Domain explored — moving to next domain.")
            next_cat = self._next_unexplored_category()
            if not next_cat:
                logger.info("[NavigationAgent] 🎯 All domains explored — switching to judgment/wellness phase.")
                final_category, reason = self._derive_final_category_and_reason()
                return {
                    "intent": "conclusion",
                    "category": final_category,
                    "reason": reason,
                    "text": self._generate_final_summary(final_category, reason),
                }
            return self._make_contextual_followup(next_cat)

        # 9️⃣ Default reflective fallback
        return self._make_reflective_followup(category, dominant_emotion)

    # ──────────────────────────────
    def _select_next(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Start with a random domain question."""
        first_category = random.choice(list(self.questions.keys()))
        return self._make_contextual_followup(first_category)

    def _make_reflective_followup(self, category: str, emotion: str) -> Dict[str, Any]:
        """Ask user to elaborate on emotion."""
        q_text = f"It sounds like this {emotion or 'feeling'} affects you deeply. Could you share what situations usually bring it up?"
        logger.info(f"[NavigationAgent] 🔁 Reflective follow-up → {q_text}")
        return {
            "id": f"reflect_{category}_{random.randint(100,999)}",
            "category": category,
            "text": q_text,
            "intent": "reflective",
            "options": []
        }

    def _make_contextual_followup(self, category: str) -> Dict[str, Any]:
        """Pick an unused question from question_pool.json."""
        questions: List[str] = self.questions.get(category, [])
        if not questions:
            logger.warning(f"[NavigationAgent] ⚠️ No questions left for category={category}.")
            return self._make_reflective_followup(category, "feeling")

        question_text = random.choice(questions)
        self.questions[category].remove(question_text)  # avoid repeats
        logger.info(f"[NavigationAgent] 🧭 Next question ({category}): {question_text}")

        return {
            "id": f"{category}_{random.randint(100,999)}",
            "category": category,
            "text": question_text,
            "intent": "diagnostic",
            "options": []
        }

    def _detect_category(self, context: Dict[str, Any], user_response: str) -> str:
        """Infer conversation domain based on user input."""
        text = (user_response or "").lower()
        if any(k in text for k in ["father", "mother", "parent", "family"]): return "family"
        if any(k in text for k in ["friend", "classmate", "peer"]): return "friends"
        if any(k in text for k in ["teacher", "professor", "school"]): return "teachers"
        if any(k in text for k in ["neighbor", "community", "society"]): return "neighbours"
        if any(k in text for k in ["myself", "me", "self", "alone"]): return "self"
        return context.get("target_category", "general")

    def _next_unexplored_category(self) -> Optional[str]:
        """Return the next unvisited category."""
        for cat in self.questions.keys():
            if self.domain_depth.get(cat, 0) == 0:
                return cat
        return None

    # ──────────────────────────────
    # 🔹 Added Helper Methods (fixes Pylance errors)
    # ──────────────────────────────
    def _derive_final_category_and_reason(self) -> tuple[str, str]:
        """Determine final category and possible reason from scores."""
        if not self.category_scores:
            return "general", "insufficient data for specific classification"

        # Convert to list of (category, score) pairs and use a safe max lookup
        top_pair = max(self.category_scores.items(), key=lambda kv: kv[1])
        final_category, _ = top_pair

        reason_map = {
            "depression": "persistent self-blame, hopelessness, or guilt",
            "anxiety": "recurring worries and fears affecting daily calm",
            "ocd": "repetitive intrusive thoughts and compulsive actions",
            "ptsd": "trauma-related flashbacks or avoidance patterns",
            "bipolar": "unstable mood swings between high and low states",
            "general": "mixed emotional stress patterns",
        }

        return final_category, reason_map.get(final_category, "emotional distress pattern")


    def _generate_final_summary(self, category: str, reason: str) -> str:
        """Generate empathetic concluding message with wellness transition."""
        goodbye_message = (
            f"Based on our reflection together, your responses align most closely with **{category.title()}**. "
            f"This seems connected to {reason}. "
            "Remember, this isn’t a diagnosis — it’s a reflection to help you understand your emotional landscape a little better.\n\n"
            "You’ve done really well opening up today, and it takes courage to do that. "
            "Before we wrap up, would you like to try a short wellness exercise — "
            "like mindful breathing, grounding, or relaxation — to help you feel calmer right now?\n\n"
            "🌿 You can also explore other **Zenark features**, such as:\n"
            "• Emotional journaling and self-reflection tools\n"
            "• Breathing and grounding exercises\n"
            "• Daily mood tracking\n"
            "• Guided affirmations and mindfulness sessions\n\n"
            "Would you like me to connect you to the **Wellness Agent** to begin?"
        )
        return goodbye_message

    # ──────────────────────────────
    def reset(self):
        self.domain_depth.clear()
        self.dataset.reload()
        self.questions = self.dataset.questions_by_category
        logger.info("🔄 NavigationAgent reset complete.")
