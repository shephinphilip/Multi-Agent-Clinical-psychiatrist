# nrc_emotion_analyzer.py
"""
NRC Hashtag Emotion Lexicon Integration
Provides psychological signal detection based on research-backed emotion-word associations
"""

import logging
from typing import Dict, List, Set, Tuple,Any
from collections import defaultdict
import os

logger = logging.getLogger("zenark.nrc")


class NRCEmotionAnalyzer:
    """Analyzes text using NRC Hashtag Emotion Lexicon"""
    
    def __init__(self, lexicon_path: str = "NRC-Hashtag-Emotion-Lexicon-v0.2.txt"):
        self.lexicon = self._load_lexicon(lexicon_path)
        self.emotions = set()
        self.total_words = len(self.lexicon)
        logger.info(f"✅ NRC Lexicon loaded with {self.total_words} unique words")
        logger.info(f"✅ Emotions available: {self.emotions}")
    
    def _load_lexicon(self, path: str) -> Dict[str, Dict[str, float]]:
        """
        Load NRC lexicon from file
        
        Format: emotion\tword\tscore
        Example: trust\tfaith\t1.42755387137464
        """
        lexicon = defaultdict(lambda: defaultdict(float))
        
        if not os.path.exists(path):
            logger.error(f"❌ NRC Lexicon file not found: {path}")
            logger.warning("📝 Continuing without NRC lexicon - using keyword-based detection only")
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        emotion = parts[0].strip()
                        word = parts[1].strip().lower()
                        
                        # Remove hashtags if present
                        if word.startswith('#'):
                            word = word[1:]
                        
                        try:
                            score = float(parts[2])
                            lexicon[word][emotion] = score
                            self.emotions.add(emotion)
                            line_count += 1
                        except ValueError:
                            continue
            
            logger.info(f"📖 Loaded {line_count} emotion-word associations")
            logger.info(f"🎭 Emotions: {sorted(self.emotions)}")
            return dict(lexicon)
        
        except Exception as e:
            logger.error(f"❌ Error loading NRC Lexicon: {e}")
            return {}
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text and return emotion scores
        
        Returns:
            Dict mapping emotion -> normalized score (0-1 range)
        """
        if not self.lexicon:
            return {}
        
        # Tokenize and clean
        words = text.lower().split()
        emotion_scores = defaultdict(float)
        matched_words = 0
        
        for word in words:
            # Remove punctuation and special characters
            word = ''.join(c for c in word if c.isalnum() or c == '_')
            
            # Remove hashtags
            if word.startswith('#'):
                word = word[1:]
            
            if word in self.lexicon:
                matched_words += 1
                for emotion, score in self.lexicon[word].items():
                    emotion_scores[emotion] += score
        
        # Normalize by number of matched words
        if matched_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= matched_words
        
        # Further normalize to 0-1 range (scores in lexicon are typically 0.5-2.0)
        max_possible = 2.0
        for emotion in emotion_scores:
            emotion_scores[emotion] = min(emotion_scores[emotion] / max_possible, 1.0)
        
        return dict(emotion_scores)
    
    def get_dominant_emotions(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N dominant emotions from text
        
        Returns:
            List of (emotion, score) tuples, sorted by score descending
        """
        scores = self.analyze_text(text)
        if not scores:
            return []
        
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_n]
    
    def detect_clinical_signals(self, text: str) -> Dict[str, float]:
        """
        Map NRC emotions to clinical mental health categories
        
        Clinical Categories:
        - depression: Linked to sadness, negative emotions, low trust
        - anxiety: Linked to fear, anticipation, surprise
        - trauma: Linked to fear, sadness, negative emotions
        - anger: Linked to anger, disgust
        - wellness: Linked to joy, trust, positive emotions
        
        Returns:
            Dict mapping clinical_category -> score (0-1)
        """
        emotion_scores = self.analyze_text(text)
        
        if not emotion_scores:
            return {
                "depression": 0.0,
                "anxiety": 0.0,
                "trauma": 0.0,
                "anger": 0.0,
                "wellness": 0.0
            }
        
        # Clinical mapping with weights
        clinical_mapping = {
            "depression": {
                "sadness": 1.0,
                "negative": 0.8,
                "fear": 0.3,
                "disgust": 0.4
            },
            "anxiety": {
                "fear": 1.0,
                "anticipation": 0.7,
                "surprise": 0.5,
                "negative": 0.3
            },
            "trauma": {
                "fear": 0.9,
                "sadness": 0.7,
                "negative": 0.6,
                "disgust": 0.4
            },
            "anger": {
                "anger": 1.0,
                "disgust": 0.7,
                "negative": 0.4
            },
            "wellness": {
                "joy": 1.0,
                "trust": 0.8,
                "positive": 0.9,
                "anticipation": 0.3
            }
        }
        
        clinical_scores = {}
        
        for category, emotion_weights in clinical_mapping.items():
            total_score = 0.0
            total_weight = 0.0
            
            for emotion, weight in emotion_weights.items():
                if emotion in emotion_scores:
                    total_score += emotion_scores[emotion] * weight
                    total_weight += weight
            
            # Normalize by total possible weight
            if total_weight > 0:
                clinical_scores[category] = total_score / total_weight
            else:
                clinical_scores[category] = 0.0
        
        return clinical_scores
    
    def get_emotion_intensity(self, text: str) -> float:
        """
        Get overall emotional intensity (0-1)
        Higher = more emotional language
        """
        emotion_scores = self.analyze_text(text)
        if not emotion_scores:
            return 0.0
        
        # Average of all emotion scores
        return sum(emotion_scores.values()) / len(emotion_scores)
    
    def is_emotionally_charged(self, text: str, threshold: float = 0.4) -> bool:
        """Check if text is emotionally charged above threshold"""
        return self.get_emotion_intensity(text) > threshold
    
    def get_emotion_summary(self, text: str) -> Dict[str, Any]:  # Changed 'any' to 'Any'
        """
        Get comprehensive emotion summary
        
        Returns:
            {
                "dominant_emotions": List[(emotion, score)],
                "clinical_signals": Dict[category -> score],
                "intensity": float,
                "is_charged": bool,
                "matched_words": List[str]
            }
        """
        emotion_scores = self.analyze_text(text)
        dominant = self.get_dominant_emotions(text, top_n=3)
        clinical = self.detect_clinical_signals(text)
        intensity = self.get_emotion_intensity(text)
        
        # Find matched words
        words = text.lower().split()
        matched = []
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word.startswith('#'):
                word = word[1:]
            if word in self.lexicon:
                matched.append(word)
        
        return {
            "dominant_emotions": dominant,
            "clinical_signals": clinical,
            "intensity": intensity,
            "is_charged": intensity > 0.4,
            "matched_words": matched,
            "all_emotions": emotion_scores
        }


# ──────────────────────────────
# TESTING
# ──────────────────────────────

if __name__ == "__main__":
    print("=== Testing NRC Emotion Analyzer ===\n")
    
    analyzer = NRCEmotionAnalyzer()
    
    if not analyzer.lexicon:
        print("⚠️ Lexicon not loaded. Please check file path.")
        exit(1)
    
    # Test texts
    test_texts = [
        "I feel so sad and hopeless, nothing makes me happy anymore",
        "I'm really worried and scared about everything",
        "I'm doing great! Life is wonderful and I feel amazing",
        "I hate everything and everyone makes me so angry",
        "My teacher is strict and I'm stressed about homework"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text}")
        print(f"{'='*60}")
        
        summary = analyzer.get_emotion_summary(text)
        
        print(f"\n🎭 Dominant Emotions:")
        for emotion, score in summary["dominant_emotions"]:
            print(f"   {emotion}: {score:.3f}")
        
        print(f"\n🏥 Clinical Signals:")
        for category, score in sorted(summary["clinical_signals"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"   {category:.<15} {score:.3f} {bar}")
        
        print(f"\n📊 Intensity: {summary['intensity']:.3f}")
        print(f"⚡ Emotionally Charged: {summary['is_charged']}")
        print(f"🔍 Matched Words: {', '.join(summary['matched_words']) if summary['matched_words'] else 'None'}")
    
    print("\n" + "="*60)
    print("=== Test Complete ===")