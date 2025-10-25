"""
utils/nrc_emotion_analyzer.py
──────────────────────────────
NRC Hashtag Emotion Lexicon Integration for Zenark

Provides psychological signal detection based on research-backed emotion-word associations.
Used internally by EmotionAgent and EmpathyAgent.
"""

import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import os

logger = logging.getLogger("zenark.utils.nrc")


class NRCEmotionAnalyzer:
    """Analyzes text using the NRC Hashtag Emotion Lexicon (v0.2)."""

    def __init__(self, lexicon_path: str = "dataset/NRC-Hashtag-Emotion-Lexicon-v0.2.txt"):
        self.emotions = set()  # ✅ define before loading lexicon
        self.lexicon = self._load_lexicon(lexicon_path)
        logger.info(f"✅ Loaded NRC Lexicon with {len(self.lexicon)} words and {len(self.emotions)} emotions.")

    def _load_lexicon(self, path: str) -> Dict[str, Dict[str, float]]:
        """Load NRC lexicon from file. Format: emotion\\tword\\tscore"""
        lexicon = defaultdict(lambda: defaultdict(float))
        if not os.path.exists(path):
            logger.error(f"❌ Lexicon not found: {path}")
            return {}

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                emotion, word, score = parts[0], parts[1].lower(), float(parts[2])
                word = word[1:] if word.startswith("#") else word
                lexicon[word][emotion] = score
                self.emotions.add(emotion)
        return dict(lexicon)

    # ──────────────────────────────
    # EMOTION DETECTION
    # ──────────────────────────────
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Compute normalized emotion scores for given text."""
        if not self.lexicon or not text:
            return {}

        words = text.lower().split()
        emotion_scores = defaultdict(float)
        matched_words = 0

        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word in self.lexicon:
                matched_words += 1
                for emo, score in self.lexicon[word].items():
                    emotion_scores[emo] += score

        if matched_words == 0:
            return {}

        for emo in emotion_scores:
            emotion_scores[emo] = min(emotion_scores[emo] / matched_words / 2.0, 1.0)
        return dict(emotion_scores)

    def get_dominant_emotions(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        scores = self.analyze_text(text)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ──────────────────────────────
    # CLINICAL SIGNAL MAPPING
    # ──────────────────────────────
    def detect_clinical_signals(self, text: str) -> Dict[str, float]:
        """Map NRC emotions to clinical categories like depression/anxiety."""
        emotion_scores = self.analyze_text(text)
        if not emotion_scores:
            return {k: 0.0 for k in ["depression", "anxiety", "trauma", "anger", "wellness"]}

        mapping = {
            "depression": {"sadness": 1.0, "negative": 0.8, "fear": 0.3},
            "anxiety": {"fear": 1.0, "anticipation": 0.7, "surprise": 0.5},
            "trauma": {"fear": 0.9, "sadness": 0.7, "disgust": 0.4},
            "anger": {"anger": 1.0, "disgust": 0.7},
            "wellness": {"joy": 1.0, "trust": 0.8, "positive": 0.9},
        }

        clinical = {}
        for cat, emo_map in mapping.items():
            total, weight = 0.0, 0.0
            for emo, w in emo_map.items():
                if emo in emotion_scores:
                    total += emotion_scores[emo] * w
                    weight += w
            clinical[cat] = round(total / weight, 3) if weight else 0.0
        return clinical

    def get_emotion_summary(self, text: str) -> Dict[str, Any]:
        """Return full structured emotion summary."""
        dominant = self.get_dominant_emotions(text)
        clinical = self.detect_clinical_signals(text)
        scores = self.analyze_text(text)
        intensity = sum(scores.values()) / len(scores) if scores else 0.0

        return {
            "dominant_emotions": dominant,
            "clinical_signals": clinical,
            "all_emotions": scores,
            "intensity": round(intensity, 3),
            "is_charged": intensity > 0.4,
            "matched_words": [w for w in text.lower().split() if w.strip("#.,!?") in self.lexicon],
        }
