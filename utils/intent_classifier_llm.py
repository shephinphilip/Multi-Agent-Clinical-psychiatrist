"""
utils/intent_classifier_llm.py
──────────────────────────────
LLM-based intent classifier for Zenark.
Uses OpenAI embeddings to classify user intent dynamically
based on similarity with examples in Intent.json.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from openai import OpenAI

client = OpenAI()


class LLMIntentClassifier:
    def __init__(self, intent_file: str):
        with open(intent_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Structure: { intent_name: [example_utterances...] }
        self.intents: Dict[str, List[str]] = {
            item["intent"]: item["text"] for item in data.get("intents", [])
        }

        # Precompute embeddings for all intent examples
        self.embeddings: Dict[str, np.ndarray] = {
            intent: self._embed_texts(texts)
            for intent, texts in self.intents.items()
        }

    # ──────────────────────────────
    # Embedding helper
    # ──────────────────────────────
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embedding matrix for list of texts."""
        if not texts:
            return np.zeros((1, 1536))  # fallback for empty intents
        response = client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return np.array([d.embedding for d in response.data])

    # ──────────────────────────────
    # Classification
    # ──────────────────────────────
    def classify(self, user_input: str) -> Tuple[str, float]:
        """
        Classify a user input into one of the known intents.

        Returns:
            (best_intent, confidence_score)
        """
        if not user_input or not isinstance(user_input, str):
            return "unknown", 0.0

        # Get embedding for the user input
        user_emb = self._embed_texts([user_input])[0]

        scores: Dict[str, float] = {}

        # Compute cosine similarity for each intent
        for intent, emb_matrix in self.embeddings.items():
            norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(user_emb)
            if np.any(norms == 0):
                scores[intent] = 0.0
            else:
                sims = np.dot(emb_matrix, user_emb) / norms
                scores[intent] = float(np.mean(sims))

        # Safely choose the intent with max similarity
        if not scores:
            return "unknown", 0.0

        best_intent: str = max(scores.keys(), key=lambda k: scores[k])  # ✅ Pylance-friendly
        best_score: float = scores[best_intent]

        return best_intent, best_score



