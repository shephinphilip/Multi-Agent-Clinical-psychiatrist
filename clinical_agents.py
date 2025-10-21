# clinical_agents.py
"""
Multi-Agent Clinical Reasoning System for Zenark
Agents collaborate to provide intelligent, empathetic mental health assessment
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from nrc_emotion_analyzer import NRCEmotionAnalyzer
from question_pool import question_pool

logger = logging.getLogger("zenark.agents")

# Load environment variables from .env file
load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────

class ConversationPhase(str, Enum):
    """Current phase of the conversation"""
    INITIAL_GREETING = "initial_greeting"
    RAPPORT_BUILDING = "rapport_building"
    GENERAL_EXPLORATION = "general_exploration"
    CLINICAL_SCREENING = "clinical_screening"
    DEEP_ASSESSMENT = "deep_assessment"
    CONCLUSION = "conclusion"
    WELLNESS_CHAT = "wellness_chat"


class ClinicalCategory(str, Enum):
    """Mental health categories"""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    PTSD = "ptsd"
    OCD = "ocd"
    BIPOLAR = "bipolar"
    WELLNESS = "wellness"
    UNKNOWN = "unknown"

@dataclass
class ContextualMemory:
    """Rich memory of user's situation"""
    # Personal details
    name: str = ""
    age_estimate: Optional[str] = None
    role: Optional[str] = None  # student, parent, etc.
    
    # Key concerns (with context)
    concerns: List[Dict[str, Any]] = field(default_factory=list)
    # Example: {"topic": "math", "emotion": "scared", "reason": "strict teacher", "turn": 3}
    
    # Support system
    supportive_people: List[str] = field(default_factory=list)
    unsupportive_people: List[str] = field(default_factory=list)
    
    # Activities & coping
    activities: List[str] = field(default_factory=list)
    coping_mechanisms: List[str] = field(default_factory=list)
    
    # Stressors
    stressors: Dict[str, List[str]] = field(default_factory=dict)
    # Example: {"academic": ["upcoming interview", "low math grades"], "family": ["strict parents"]}
    
    # Timeline of emotions
    emotion_timeline: List[Dict] = field(default_factory=list)
    # Example: {"turn": 5, "emotion": "tired", "frequency": "most of the time"}



@dataclass
class ConversationContext:
    """Maintains conversation state and history"""
    username: str = "User"
    phase: ConversationPhase = ConversationPhase.INITIAL_GREETING
    turn_count: int = 0
    
    # User profile
    age_band: Optional[str] = None
    role: Optional[str] = None
    
    # ENHANCED: Contextual Memory
    memory: ContextualMemory = field(default_factory=ContextualMemory)  # NEW
    
    # Clinical signals
    mentioned_topics: List[str] = field(default_factory=list)
    emotional_signals: Dict[str, float] = field(default_factory=dict)
    clinical_signals: Dict[str, float] = field(default_factory=dict)
    
    # Assessment tracking
    categories_explored: Dict[str, int] = field(default_factory=lambda: {
        "depression": 0,
        "anxiety": 0,
        "ptsd": 0,
        "ocd": 0,
        "bipolar": 0
    })
    category_scores: Dict[str, float] = field(default_factory=lambda: {
        "depression": 0.0,
        "anxiety": 0.0,
        "ptsd": 0.0,
        "ocd": 0.0,
        "bipolar": 0.0
    })
    
    # Question tracking
    asked_questions: List[str] = field(default_factory=list)
    answered_questions: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation memory
    last_bot_message: Optional[str] = None
    last_user_message: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # State flags
    shows_distress: bool = False
    is_wellness_confirmed: bool = False
    needs_deeper_exploration: bool = False
    ready_to_conclude: bool = False
    concluded: bool = False
    
    # Reasoning
    current_hypothesis: Optional[str] = None
    confidence_in_hypothesis: float = 0.0


# ──────────────────────────────
# SIGNAL DETECTOR AGENT
# ──────────────────────────────

class SignalDetectorAgent:
    """
    Detects psychological and emotional signals from user responses
    Uses NRC emotion lexicon + keyword analysis
    """
    
    def __init__(self):
        self.nrc = NRCEmotionAnalyzer()
        
        # Clinical keyword patterns
        self.distress_keywords = {
            "depression": [
                "sad", "hopeless", "worthless", "empty", "numb", "crying", "suicide",
                "die", "death", "kill myself", "end it", "no point", "give up",
                "tired all the time", "can't sleep", "no energy", "don't care",
                "hate myself", "failure", "useless", "burden"
            ],
            "anxiety": [
                "worried", "anxious", "scared", "nervous", "panic", "fear", "afraid",
                "can't relax", "tense", "on edge", "overwhelmed", "stressed",
                "racing thoughts", "can't stop thinking", "what if", "worried about"
            ],
            "trauma": [
                "flashback", "nightmare", "abuse", "hurt me", "scared of", "traumatized",
                "can't forget", "haunted by", "happened to me", "unsafe"
            ],
            "ocd": [
                "keep checking", "can't stop", "over and over", "have to", "ritual",
                "contaminated", "germs", "counting", "arrange", "order", "perfect"
            ],
            "anger": [
                "angry", "mad", "hate", "furious", "rage", "annoyed", "irritated",
                "fight", "hit", "yell", "scream", "lose control"
            ]
        }
        
        # Wellness indicators
        self.wellness_keywords = [
            "good", "great", "fine", "okay", "happy", "enjoy", "fun", "excited",
            "love", "awesome", "wonderful", "better", "improving"
        ]
        
        # School/life topics
        self.life_topics = {
            "school": ["school", "class", "teacher", "homework", "assignment", "test", "exam", "grade", "study"],
            "friends": ["friend", "buddy", "peer", "classmate", "hanging out", "play with"],
            "family": ["mom", "dad", "mother", "father", "parent", "sibling", "brother", "sister", "home"],
            "activities": ["sport", "game", "play", "hobby", "activity", "cricket", "football", "music"],
            "academic": ["math", "science", "english", "subject", "lesson", "learning"]
        }
    
    def analyze_response(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Comprehensive analysis of user response
        
        Returns:
            {
                "distress_detected": bool,
                "distress_level": float (0-1),
                "clinical_signals": Dict[category -> score],
                "emotional_tone": Dict[emotion -> score],
                "topics_mentioned": List[str],
                "wellness_indicators": float (0-1),
                "needs_followup": bool,
                "suggested_focus": Optional[str]
            }
        """
        text_lower = text.lower()
        
        # NRC emotion analysis
        emotional_tone = self.nrc.analyze_text(text)
        clinical_signals_nrc = self.nrc.detect_clinical_signals(text)
        
        # Keyword-based detection
        clinical_signals_keywords = self._detect_clinical_keywords(text_lower)
        
        # Merge signals (NRC + keywords)
        clinical_signals = {}
        for category in ["depression", "anxiety", "trauma", "anger"]:
            nrc_score = clinical_signals_nrc.get(category, 0.0)
            keyword_score = clinical_signals_keywords.get(category, 0.0)
            # Weighted combination (NRC 40%, keywords 60% because keywords are more direct)
            clinical_signals[category] = (nrc_score * 0.4) + (keyword_score * 0.6)
        
        # Detect topics
        topics_mentioned = self._detect_topics(text_lower)
        
        # Wellness indicators
        wellness_score = self._calculate_wellness_score(text_lower, emotional_tone)
        
        # Overall distress
        max_clinical_signal = max(clinical_signals.values()) if clinical_signals else 0.0
        distress_detected = max_clinical_signal > 0.3 or self._has_crisis_language(text_lower)
        distress_level = max_clinical_signal
        
        # Determine if needs follow-up
        needs_followup = distress_detected or len(topics_mentioned) > 0
        
        # Suggest focus area
        suggested_focus = None
        if distress_detected:
            suggested_focus = max(clinical_signals.items(), key=lambda x: x[1])[0]
        elif topics_mentioned:
            suggested_focus = topics_mentioned[0]
        
        return {
            "distress_detected": distress_detected,
            "distress_level": distress_level,
            "clinical_signals": clinical_signals,
            "emotional_tone": emotional_tone,
            "topics_mentioned": topics_mentioned,
            "wellness_indicators": wellness_score,
            "needs_followup": needs_followup,
            "suggested_focus": suggested_focus
        }
    
    def _detect_clinical_keywords(self, text: str) -> Dict[str, float]:
        """Detect clinical keywords and return scores"""
        scores = {}
        
        for category, keywords in self.distress_keywords.items():
            matches = sum(1 for kw in keywords if kw in text)
            
            # ✅ CHANGE: More aggressive scoring
            # Before: scores[category] = min(matches / 3.0, 1.0)
            # After: Even 1 keyword match should register
            if matches > 0:
                scores[category] = min(matches / 2.0, 1.0)  # Changed from /3.0 to /2.0
            else:
                scores[category] = 0.0
        
        # ✅ ADD: Extra boost for serious keywords
        serious_trauma_keywords = ["beat", "hit", "hurt", "slap", "punch", "abuse", "violence"]
        if any(kw in text for kw in serious_trauma_keywords):
            scores["trauma"] = max(scores.get("trauma", 0.0), 0.8)  # Minimum 0.8 for physical violence
        
        # ✅ ADD: Boost for negative expressions
        negative_phrases = ["not good at all", "terrible", "awful", "horrible", "worst"]
        if any(phrase in text for phrase in negative_phrases):
            scores["depression"] = max(scores.get("depression", 0.0), 0.4)
        
        return scores
        
    def _detect_topics(self, text: str) -> List[str]:
        """Detect life topics mentioned"""
        topics = []
        
        for topic, keywords in self.life_topics.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _calculate_wellness_score(self, text: str, emotional_tone: Dict) -> float:
        """Calculate wellness indicators"""
        # Keyword-based
        wellness_keyword_count = sum(1 for kw in self.wellness_keywords if kw in text)
        keyword_score = min(wellness_keyword_count / 3.0, 1.0)
        
        # Emotion-based (positive emotions from NRC)
        emotion_score = emotional_tone.get("joy", 0.0) + emotional_tone.get("positive", 0.0)
        emotion_score = min(emotion_score, 1.0)
        
        # Combined
        return (keyword_score * 0.5) + (emotion_score * 0.5)
    
    def _has_crisis_language(self, text: str) -> bool:
        """Detect immediate crisis language"""
        crisis_patterns = [
            "kill myself", "end my life", "suicide", "want to die", "better off dead",
            "hurt myself", "self harm", "end it all"
        ]
        return any(pattern in text for pattern in crisis_patterns)


# ──────────────────────────────
# CLINICAL REASONER AGENT
# ──────────────────────────────

class ClinicalReasonerAgent:
    """
    Makes high-level decisions about the assessment
    - Determines conversation phase
    - Decides when to screen vs explore
    - Identifies when to conclude
    """
    
    def __init__(self):
        self.phase_thresholds = {
            "move_to_screening": 3,  # turns of rapport building
            "move_to_deep": 5,       # screening questions before deep dive
            "min_turns_to_conclude": 8,
            "wellness_confirmation_turns": 5
        }
    
    def reason_next_step(self, context: ConversationContext, signal_analysis: Dict) -> Dict[str, Any]:
        """
        Main reasoning function - determines what to do next
        
        Returns:
            {
                "action": str (explore_topic, screen_category, deep_assess, conclude, wellness_chat),
                "target": Optional[str] (category or topic to focus on),
                "phase_transition": Optional[ConversationPhase],
                "reasoning": str (explanation of decision)
            }
        """
        logger.info(f"🧠 REASONER: Phase={context.phase.value}, Turn={context.turn_count}")
        logger.info(f"🧠 Clinical signals: {signal_analysis['clinical_signals']}")
        logger.info(f"🧠 Topics: {signal_analysis['topics_mentioned']}")
        
        # Crisis detected - immediate conclusion
        if signal_analysis["distress_level"] > 0.8:
            return {
                "action": "conclude",
                "target": "crisis",
                "phase_transition": ConversationPhase.CONCLUSION,
                "reasoning": "High distress detected - immediate intervention needed"
            }
        
        # Phase-specific reasoning
        if context.phase == ConversationPhase.INITIAL_GREETING:
            return self._reason_initial_greeting(context)
        
        elif context.phase == ConversationPhase.RAPPORT_BUILDING:
            return self._reason_rapport_building(context, signal_analysis)  # ✅ This should be called
        
        elif context.phase == ConversationPhase.GENERAL_EXPLORATION:
            return self._reason_general_exploration(context, signal_analysis)
        
        elif context.phase == ConversationPhase.CLINICAL_SCREENING:
            return self._reason_clinical_screening(context, signal_analysis)
        
        elif context.phase == ConversationPhase.DEEP_ASSESSMENT:
            return self._reason_deep_assessment(context, signal_analysis)
        
        elif context.phase == ConversationPhase.WELLNESS_CHAT:
            return self._reason_wellness_chat(context, signal_analysis)
        
        else:
            # Fallback - should never reach here
            return {
                "action": "explore_general",
                "target": "general",
                "phase_transition": None,
                "reasoning": "Default exploration (unknown phase)"
            }
        
    def _reason_initial_greeting(self, context: ConversationContext) -> Dict:
        """First interaction - establish rapport"""
        return {
            "action": "greet",
            "target": None,
            "phase_transition": ConversationPhase.RAPPORT_BUILDING,
            "reasoning": "Initial greeting and rapport establishment"
        }
    
    def _reason_rapport_building(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
        """Build trust and understand user state"""
        
        memory = getattr(context, 'memory', None)
    
        # ✅ Priority 0: If this is the FIRST response after greeting, always acknowledge what they said
        if context.turn_count == 1:
            # User just responded to our greeting - acknowledge their response
            if signal_analysis["distress_detected"] or signal_analysis.get("distress_level", 0) > 0.2:
                # They mentioned something distressing right away
                return {
                    "action": "explore_concern",
                    "target": signal_analysis.get("suggested_focus", "general"),
                    "phase_transition": None,
                    "reasoning": "User expressed concern in initial response - exploring empathetically"
                }
            elif signal_analysis["topics_mentioned"]:
                # They mentioned a specific topic
                return {
                    "action": "explore_topic",
                    "target": signal_analysis["topics_mentioned"][0],
                    "phase_transition": None,
                    "reasoning": f"User mentioned {signal_analysis['topics_mentioned'][0]} - following up"
                }
            else:
                # Generic first response
                return {
                    "action": "explore_general",
                    "target": "wellbeing",
                    "phase_transition": None,
                    "reasoning": "Acknowledging user's first response and exploring"
                }
            
        # Priority 1: If user mentioned a concern, explore it FIRST
        if memory and memory.concerns:
            latest_concern = memory.concerns[-1]
            
            if not self._has_followed_up(latest_concern, context):
                return {
                    "action": "explore_concern",
                    "target": latest_concern,
                    "phase_transition": ConversationPhase.GENERAL_EXPLORATION,
                    "reasoning": f"User expressed concern about {latest_concern.get('topic')} - exploring deeper"
                }
        
        # If distress detected early, move to screening
        if signal_analysis["distress_detected"]:
            return {
                "action": "screen_category",
                "target": signal_analysis.get("suggested_focus", "general"),
                "phase_transition": ConversationPhase.CLINICAL_SCREENING,
                "reasoning": "Distress detected - initiating clinical screening"
            }
        
        # If topics mentioned, explore them
        if signal_analysis["topics_mentioned"]:
            return {
                "action": "explore_topic",
                "target": signal_analysis["topics_mentioned"][0],
                "phase_transition": ConversationPhase.GENERAL_EXPLORATION if context.turn_count >= 2 else None,
                "reasoning": f"Following up on mentioned topic: {signal_analysis['topics_mentioned'][0]}"
            }
        
        # Continue building rapport
        if context.turn_count < self.phase_thresholds["move_to_screening"]:
            return {
                "action": "explore_general",
                "target": "life",
                "phase_transition": None,
                "reasoning": "Building rapport through general conversation"
            }
        
        # Move to exploration
        return {
            "action": "explore_general",
            "target": "wellbeing",
            "phase_transition": ConversationPhase.GENERAL_EXPLORATION,
            "reasoning": "Transitioning to general wellbeing exploration"
        }
    
    def _reason_general_exploration(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
        """Explore life areas to detect issues"""
        
        memory = getattr(context, 'memory', None)
        
        # Priority: If user mentioned SERIOUS concern (teacher violence), explore it deeply first
        if memory and memory.concerns:
            teacher_violence_concerns = [
                c for c in memory.concerns 
                if c.get('topic') == 'teacher violence' and c.get('severity') == 'high'
            ]
            
            # If we have a high-severity concern about teacher violence, explore it
            for concern in teacher_violence_concerns:
                if not self._has_followed_up(concern, context) or context.turn_count < 5:
                    return {
                        "action": "explore_concern",
                        "target": concern,
                        "phase_transition": None,
                        "reasoning": "User mentioned teacher violence - need deeper exploration before screening"
                    }
        
        # If distress emerges, move to screening
        if signal_analysis["distress_detected"]:
            return {
                "action": "screen_category",
                "target": signal_analysis.get("suggested_focus", "general"),
                "phase_transition": ConversationPhase.CLINICAL_SCREENING,
                "reasoning": "Distress emerged - moving to clinical screening"
            }
        
        # Explore all relevant areas first before screening
        unexplored_topics = [t for t in ["teachers", "homework", "friends", "family", "activities"] 
                            if t not in context.mentioned_topics]
        
        if unexplored_topics:
            return {
                "action": "explore_topic",
                "target": unexplored_topics[0],
                "phase_transition": None,
                "reasoning": f"Exploring life area: {unexplored_topics[0]}"
            }
        
        # Only move to screening if we've explored enough
        if context.turn_count >= 5:
            return {
                "action": "screen_category",
                "target": "first_round",
                "phase_transition": ConversationPhase.CLINICAL_SCREENING,
                "reasoning": "Multiple topics explored - beginning systematic screening"
            }
        
        # Not ready for screening yet - continue exploring
        return {
            "action": "explore_general",
            "target": "coping",
            "phase_transition": None,
            "reasoning": "Continue exploring before screening"
        }
    
    def _reason_clinical_screening(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
        """Systematic screening of mental health categories"""
        
        # Check if we've done first round screening (1 question per category)
        categories_with_questions = [cat for cat, count in context.categories_explored.items() if count > 0]
        all_categories = ["depression", "anxiety", "ptsd", "ocd", "bipolar"]
        
        # First round - ask one from each category
        if len(categories_with_questions) < len(all_categories):
            unscreened = [cat for cat in all_categories if context.categories_explored[cat] == 0]
            return {
                "action": "screen_category",
                "target": unscreened[0],
                "phase_transition": None,
                "reasoning": f"First round screening: {unscreened[0]}"
            }
        
        # Analyze first round results
        positive_categories = [cat for cat, score in context.category_scores.items() if score >= 0.5]
        
        if not positive_categories:
            # No positive screens - likely wellness
            return {
                "action": "conclude",
                "target": "wellness",
                "phase_transition": ConversationPhase.CONCLUSION,
                "reasoning": "All screening negative - wellness profile"
            }
        
        elif len(positive_categories) == 1:
            # Single category positive - deep dive
            return {
                "action": "deep_assess",
                "target": positive_categories[0],
                "phase_transition": ConversationPhase.DEEP_ASSESSMENT,
                "reasoning": f"Single positive screen: {positive_categories[0]} - initiating deep assessment"
            }
        
        else:
            # Multiple positive - second round screening
            for cat in positive_categories:
                if context.categories_explored[cat] < 2:
                    return {
                        "action": "screen_category",
                        "target": cat,
                        "phase_transition": None,
                        "reasoning": f"Second round screening for: {cat}"
                    }
            
            # After second round, pick highest scoring
            highest_cat = max(positive_categories, key=lambda c: context.category_scores[c])
            return {
                "action": "deep_assess",
                "target": highest_cat,
                "phase_transition": ConversationPhase.DEEP_ASSESSMENT,
                "reasoning": f"Multiple positive screens - focusing on highest: {highest_cat}"
            }
    
    def _reason_deep_assessment(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
        """Deep assessment of identified category"""
        
        # Get the category we're assessing
        current_category = context.current_hypothesis
        
        if not current_category:
            # Shouldn't happen, but fallback
            return {
                "action": "conclude",
                "target": "general",
                "phase_transition": ConversationPhase.CONCLUSION,
                "reasoning": "No hypothesis set - concluding"
            }
        
        # Check if we have enough questions (5-7 for deep assessment)
        questions_in_category = context.categories_explored.get(current_category, 0)
        
        if questions_in_category >= 7:
            # Sufficient data collected
            return {
                "action": "conclude",
                "target": current_category,
                "phase_transition": ConversationPhase.CONCLUSION,
                "reasoning": f"Deep assessment complete for {current_category}"
            }
        
        # Continue deep assessment
        return {
            "action": "deep_assess",
            "target": current_category,
            "phase_transition": None,
            "reasoning": f"Continuing deep assessment: {current_category} (q#{questions_in_category + 1})"
        }
    
    def _reason_wellness_chat(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
        """Wellness-focused supportive conversation"""
        
        # If distress emerges, switch back to screening
        if signal_analysis["distress_detected"]:
            return {
                "action": "screen_category",
                "target": signal_analysis.get("suggested_focus", "general"),
                "phase_transition": ConversationPhase.CLINICAL_SCREENING,
                "reasoning": "Distress detected during wellness chat - initiating screening"
            }
        
        # Continue wellness chat for a few turns
        if context.turn_count < self.phase_thresholds["min_turns_to_conclude"]:
            # Explore different wellness topics
            wellness_topics = ["academics", "hobbies", "friends", "goals", "family"]
            unexplored = [t for t in wellness_topics if t not in context.mentioned_topics]
            
            if unexplored:
                return {
                    "action": "wellness_chat",
                    "target": unexplored[0],
                    "phase_transition": None,
                    "reasoning": f"Wellness conversation about: {unexplored[0]}"
                }
        
        # Ready to conclude wellness conversation
        return {
            "action": "conclude",
            "target": "wellness",
            "phase_transition": ConversationPhase.CONCLUSION,
            "reasoning": "Wellness conversation complete - concluding positively"
        }
    
    def _has_followed_up(self, concern: Dict, context: ConversationContext) -> bool:
        """Check if we've followed up on this concern"""
        topic = concern.get('topic', '').lower()
        if not topic:
            return False
        
        # Check if we've asked about this in last 3 messages
        recent = context.conversation_history[-3:] if len(context.conversation_history) >= 3 else context.conversation_history
        
        for msg in recent:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').lower()
                # Check if topic or related keywords are in our response
                if topic in content:
                    return True
                # Check for related terms
                if topic == 'sadness' and any(word in content for word in ['sad', 'feeling', 'mood']):
                    return True
                if topic == 'exams' and any(word in content for word in ['exam', 'test', 'study']):
                    return True
        
        return False
        
    def _has_explored_stressor(self, category: str, context: ConversationContext) -> bool:
        """
        Check if we've already explored this stressor category
        
        Args:
            category: Stressor category (academic, family, social, professional)
            context: Conversation context
        
        Returns:
            True if we've already explored this category
        """
        # Check recent bot messages
        recent_messages = context.conversation_history[-5:] if len(context.conversation_history) >= 5 else context.conversation_history
        
        # Keywords associated with each category
        category_keywords = {
            'academic': ['school', 'class', 'teacher', 'homework', 'test', 'exam', 'grade', 'study', 'assignment'],
            'family': ['family', 'parent', 'mom', 'dad', 'mother', 'father', 'home', 'sibling'],
            'social': ['friend', 'peer', 'classmate', 'social', 'relationship'],
            'professional': ['work', 'job', 'interview', 'career', 'workplace']
        }
        
        keywords = category_keywords.get(category, [category])
        
        for msg in recent_messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').lower()
                # If we've asked about this category keywords, we've explored it
                if any(keyword in content for keyword in keywords):
                    return True
        
        return False

# ──────────────────────────────
# QUESTION SELECTOR AGENT
# ──────────────────────────────

class QuestionSelectorAgent:
    """
    Selects appropriate clinical questions from question_pool
    Implements adaptive narrowing algorithm
    """
    
    def __init__(self):
        self.question_pool = question_pool
        self.category_to_pool = self._build_category_index()
        logger.info(f"✅ Question pool loaded: {len(self.question_pool)} questions")
    
    def _build_category_index(self) -> Dict[str, List[Dict]]:
        """Index questions by category"""
        index = {}
        for q in self.question_pool:
            cat = q.get("category", "unknown")
            if cat not in index:
                index[cat] = []
            index[cat].append(q)
        return index
    
    def select_question(self, category: str, context: ConversationContext) -> Optional[Dict]:
        """
        Select next question from specified category
        
        Args:
            category: Clinical category (depression, anxiety, etc.)
            context: Conversation context with history
        
        Returns:
            Question dict or None if no suitable question
        """
        if category not in self.category_to_pool:
            logger.warning(f"Category {category} not in pool")
            return None
        
        available_questions = [
            q for q in self.category_to_pool[category]
            if q["id"] not in context.asked_questions
        ]
        
        if not available_questions:
            logger.warning(f"No more questions available for {category}")
            return None
        
        # For first round, prefer questions from diverse instruments
        if context.categories_explored[category] == 0:
            # Get unique instruments
            instruments = list(set(q.get("instrument") for q in available_questions))
            if instruments:
                # Pick first question from first instrument
                target_instrument = instruments[0]
                instrument_questions = [q for q in available_questions if q.get("instrument") == target_instrument]
                return instrument_questions[0] if instrument_questions else available_questions[0]
        
        # For subsequent rounds, continue same instrument or pick new one
        return available_questions[0]
    
    def select_first_round_questions(self) -> Dict[str, Dict]:
        """
        Select first question from each category for initial screening
        
        Returns:
            Dict mapping category -> question
        """
        first_questions = {}
        
        for category in ["depression", "anxiety", "ptsd", "ocd", "bipolar"]:
            if category in self.category_to_pool and self.category_to_pool[category]:
                first_questions[category] = self.category_to_pool[category][0]
        
        return first_questions


# ──────────────────────────────
# RESPONSE SCORER AGENT
# ──────────────────────────────

class ResponseScorerAgent:
    """
    Scores user responses to clinical questions
    Maps natural language to option scores
    """
    
    def __init__(self):
        # Common response patterns
        self.response_patterns = {
            "yes": ["yes", "yeah", "yep", "yup", "definitely", "absolutely", "for sure"],
            "no": ["no", "nope", "nah", "not really", "no way", "never"],
            "sometimes": ["sometimes", "occasionally", "once in a while", "here and there"],
            "often": ["often", "frequently", "a lot", "many times", "usually"],
            "always": ["always", "all the time", "constantly", "every day"],
            "rarely": ["rarely", "hardly ever", "seldom", "almost never"],
        }
    
    def score_response(self, user_response: str, question: Dict) -> Tuple[Optional[int], str]:
        """
        Score a user response against a question
        
        Args:
            user_response: User's natural language response
            question: Question dict with options and score_range
        
        Returns:
            (score, matched_option) or (None, "") if no match
        """
        if "options" not in question or "score_range" not in question:
            return None, ""
        
        options = question["options"]
        score_range = question["score_range"]
        user_lower = user_response.lower().strip()
        
        # Method 1: Exact match
        for i, option in enumerate(options):
            if user_lower == option.lower():
                return score_range[i], option
        
        # Method 2: Numeric selection (1, 2, 3...)
        if user_lower.isdigit():
            idx = int(user_lower) - 1
            if 0 <= idx < len(score_range):
                return score_range[idx], options[idx]
        
        # Method 3: Partial text match
        for i, option in enumerate(options):
            if user_lower in option.lower() or option.lower() in user_lower:
                return score_range[i], option
        
        # Method 4: Pattern matching
        for pattern_type, patterns in self.response_patterns.items():
            if any(p in user_lower for p in patterns):
                # Try to match pattern to options
                for i, option in enumerate(options):
                    if pattern_type in option.lower() or any(p in option.lower() for p in patterns):
                        return score_range[i], option
        
        # Method 5: Yes/No questions - special handling
        if len(options) == 2 and "yes" in options[0].lower() and "no" in options[1].lower():
            if any(p in user_lower for p in self.response_patterns["yes"]):
                return score_range[0], options[0]
            elif any(p in user_lower for p in self.response_patterns["no"]):
                return score_range[1], options[1]
        
        # No match found
        logger.warning(f"Could not score response: '{user_response}' for options: {options}")
        return None, ""
    
    def calculate_category_score(self, question: Dict, score: int) -> float:
        """
        Calculate normalized category contribution
        
        Returns:
            0-1 score representing severity for that question
        """
        score_range = question.get("score_range", [0])
        max_score = max(score_range)
        min_score = min(score_range)
        
        if max_score == min_score:
            return 0.0
        
        # Normalize to 0-1
        normalized = (score - min_score) / (max_score - min_score)
        return normalized


# ──────────────────────────────
# EMPATHY AGENT
# ──────────────────────────────

class EmpathyAgent:
    """
    Generates empathetic, human-like responses
    Wraps clinical questions in compassionate language
    """
    
    def __init__(self):
        self.empathy_styles = {
            "acknowledgment": [
                "I hear you",
                "That sounds really {emotion}",
                "I can understand why you'd feel that way",
                "Thank you for sharing that with me",
                "That takes courage to talk about"
            ],
            "validation": [
                "Your feelings are completely valid",
                "It's okay to feel {emotion}",
                "Many people experience this",
                "You're not alone in feeling this way"
            ],
            "gentle_probe": [
                "Can you tell me a bit more about that?",
                "What's that been like for you?",
                "How has that been affecting you?",
                "When did you first notice this?"
            ]
        }
    
    def generate_empathetic_wrapper(self, clinical_question: str, context: ConversationContext, 
                                   signal_analysis: Optional[Dict] = None) -> str:
        """
        Wrap a clinical question in empathetic, natural language
        
        Args:
            clinical_question: The formal clinical question text
            context: Conversation context
            signal_analysis: Optional analysis of previous user response
        
        Returns:
            Natural, empathetic question
        """
        # If first question in category, add gentle intro
        category = None
        if context.current_hypothesis:
            category = context.current_hypothesis
        
        intro = self._get_category_intro(category, context)
        
        # Add acknowledgment if user showed distress
        acknowledgment = ""
        if signal_analysis and signal_analysis.get("distress_detected"):
            emotion = self._identify_emotion(signal_analysis)
            acknowledgment = f"I can see that's been {emotion} for you. "
        
        # Simplify clinical language
        simplified = self._simplify_clinical_language(clinical_question)
        
        # Combine
        if intro:
            return f"{intro} {acknowledgment}{simplified}"
        elif acknowledgment:
            return f"{acknowledgment}{simplified}"
        else:
            return simplified
    
    def _get_category_intro(self, category: Optional[str], context: ConversationContext) -> str:
        """Get intro based on category (only for first question)"""
        if not category or category in context.mentioned_topics:
            return ""
        
        intros = {
            "depression": "I'd like to understand how you've been feeling emotionally.",
            "anxiety": "Let's talk about worry and nervousness for a moment.",
            "ptsd": "I want to check in about any difficult experiences you might have had.",
            "ocd": "Can we explore some thoughts and behaviors you might experience?",
            "bipolar": "I'd like to ask about changes in your mood and energy."
        }
        
        context.mentioned_topics.append(category)
        return intros.get(category, "")
    
    def _simplify_clinical_language(self, text: str) -> str:
        """Make clinical language more conversational"""
        # Remove formal phrasing
        text = text.replace("In the past week, have you", "Have you been")
        text = text.replace("During the past", "In the past")
        text = text.replace("experienced", "feeling")
        
        # Make it more kid-friendly if needed
        # (You can expand this based on age_band from context)
        
        return text
    
    def _identify_emotion(self, signal_analysis: Dict) -> str:
        """Identify dominant emotion from analysis"""
        emotions = signal_analysis.get("emotional_tone", {})
        if not emotions:
            return "difficult"
        
        # Map to relatable terms
        emotion_map = {
            "sadness": "tough",
            "fear": "scary",
            "anger": "frustrating",
            "disgust": "upsetting",
            "joy": "positive",
            "anticipation": "uncertain"
        }
        
        dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "difficult"
        return emotion_map.get(dominant, "challenging")
    
    def generate_conclusion_message(self, category: str, severity: str, context: ConversationContext) -> str:
        """Generate empathetic conclusion message"""
        
        if category == "wellness":
            return (
                f"Thank you for talking with me today, {context.username}. "
                f"It sounds like you're doing pretty well overall! "
                f"Remember, I'm always here if you need someone to talk to. "
                f"Keep taking care of yourself, and don't hesitate to reach out if things change. "
                f"You're doing great! 💙"
            )
        
        elif category == "crisis":
            return (
                f"{context.username}, I'm really concerned about what you've shared. "
                f"What you're feeling is very serious, and I want you to get immediate help. "
                f"Please reach out to a trusted adult, call the crisis hotline at 988, "
                f"or go to your nearest emergency room. You don't have to face this alone. "
                f"There are people who care and want to help you. 🆘"
            )
        
        else:
            # Regular category conclusion
            severity_messages = {
                "mild": (
                    f"Thank you for being so open with me today, {context.username}. "
                    f"Based on what you've shared, it sounds like you're experiencing some {category}-related concerns. "
                    f"These feelings are valid, and it's good that you're talking about them. "
                    f"I'd recommend speaking with a school counselor or trusted adult who can provide support. "
                    f"Remember, reaching out for help is a sign of strength, not weakness. 💪"
                ),
                "moderate": (
                    f"{context.username}, I really appreciate you trusting me with your feelings today. "
                    f"From our conversation, it seems like you're dealing with some significant {category} symptoms. "
                    f"I think it would be really helpful for you to talk to a mental health professional – "
                    f"like a therapist or counselor – who can give you the support you deserve. "
                    f"You don't have to go through this alone. Many people feel this way, and there's help available. "
                    f"Please reach out to a trusted adult or school counselor soon. 🤝"
                ),
                "severe": (
                    f"{context.username}, thank you for being so honest with me. I can tell you're going through a really difficult time. "
                    f"What you've shared suggests you're experiencing serious {category} symptoms, "
                    f"and I'm concerned about you. It's really important that you talk to a mental health professional soon – "
                    f"a therapist, doctor, or counselor who specializes in helping people with these feelings. "
                    f"Please let a trusted adult know what you're going through. You deserve support, "
                    f"and there are people who can help you feel better. This won't last forever. 💙"
                )
            }
            
            return severity_messages.get(severity, severity_messages["moderate"])


# ──────────────────────────────
# CONVERSATION GENERATOR AGENT
# ──────────────────────────────

class ConversationGeneratorAgent:
    """
    Generates natural, flowing conversational responses
    Handles non-assessment dialogue (greetings, topic exploration, wellness chat)
    """
    
    def __init__(self,api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or OPEN_AI_API_KEY)
        self.model = model
        self.greeting_templates = {
            "first_time": [
                "Hey there! I'm Zen, your personal companion. I'm here to listen and chat about how you're doing. "
                "This is a safe space where you can share whatever's on your mind. How are you feeling today?",
                
                "Hi! I'm Zen, and I'm really glad you're here. I'm here to understand how things are going for you. "
                "There are no right or wrong answers – just be yourself. What's been on your mind lately?",
                
                "Hello! My name is Zen, and I'm here to be a friend and listener. "
                "You can talk to me about anything – how you're feeling, what's happening in your life, whatever you'd like. "
                "So, how have things been for you?"
            ],
            "returning": [
                "Hey {name}! It's good to see you again. How have you been since we last talked?",
                
                "Hi {name}! Welcome back. I've been thinking about you. How are things going?",
                
                "{name}! Good to have you back. What's been happening in your world?"
            ]
        }
        
        self.topic_exploration = {
            "school": [
                "How's school going for you?",
                "What's your favorite subject in school?",
                "How are things with your teachers?",
                "Tell me about a typical day at school.",
                "Is there anything at school that's been bothering you?",
                "What do you enjoy most about school?",
                "How are your grades? Any subjects giving you trouble?"
            ],
            "friends": [
                "How are things with your friends?",
                "Do you have a close group of friends?",
                "What do you like to do with your friends?",
                "Have you been hanging out with friends much lately?",
                "Is there anything going on with friendships that you'd like to talk about?",
                "Who do you feel closest to?"
            ],
            "family": [
                "How are things at home with your family?",
                "Do you get along well with your family members?",
                "Is there anything happening at home you'd like to talk about?",
                "How do you spend time with your family?",
                "Do you feel supported by your family?"
            ],
            "activities": [
                "What do you like to do for fun?",
                "Do you play any sports or have hobbies?",
                "What makes you happy?",
                "When was the last time you really enjoyed yourself?",
                "Is there something you're passionate about?"
            ],
            "academics": [
                "How's the workload been at school?",
                "Are you feeling stressed about any assignments or tests?",
                "What subject are you working on right now?",
                "Do you feel like you're keeping up with your schoolwork?",
                "Have your grades changed recently?"
            ]
        }
        
        self.wellness_prompts = {
            "general": [
                "That's wonderful! What else has been making you feel good?",
                "I'm so glad to hear that! What are you most looking forward to?",
                "That sounds really positive! What's been the best part of your week?",
                "It's great that things are going well! Tell me more about what's making you happy."
            ],
            "achievements": [
                "That's amazing! You should be proud of yourself.",
                "Wow, that's a great accomplishment! How did that feel?",
                "That's fantastic! What helped you achieve that?",
                "That's something to celebrate! How are you feeling about it?"
            ],
            "support": [
                "It sounds like you have good people around you. That's so important.",
                "I'm glad you have that support. That makes a real difference.",
                "Having good friends/family is really valuable. You're lucky to have them.",
                "That support system sounds really strong. How does that feel?"
            ]
        }
    
    def generate_greeting(self, is_returning: bool, username: str = "there") -> str:
        """Generate initial greeting"""
        import random
        
        if is_returning:
            template = random.choice(self.greeting_templates["returning"])
            return template.format(name=username)
        else:
            return random.choice(self.greeting_templates["first_time"])
    
    def generate_topic_exploration(self, topic: str, context: ConversationContext) -> str:
        """Generate question to explore a life topic"""
        import random
        
        if topic not in self.topic_exploration:
            return "What's been on your mind lately?"
        
        # Get questions for this topic that haven't been asked
        available_questions = [
            q for q in self.topic_exploration[topic]
            if q not in context.conversation_history
        ]
        
        if not available_questions:
            available_questions = self.topic_exploration[topic]
        
        return random.choice(available_questions)
    

    def generate_follow_up(self, context: Dict) -> str:
        """Generate natural follow-up based on what user said"""
        user_message = context.get('user_message', '')
        conversation_history = context.get('conversation_history', [])
        username = context.get('username', '')
        empathy_level = context.get('empathy_level', 'moderate')
        memory_summary = context.get('memory_summary') or "No detailed history captured yet."
        memory_highlight = context.get('memory_highlight') or ""

        # ✅ ENHANCED: Detect serious content in user message
        text_lower = user_message.lower()
        is_serious = any(word in text_lower for word in [
            'beat', 'hit', 'hurt', 'abuse', 'violence', 'not good at all',
            'terrible', 'awful', 'horrible', 'suicide', 'kill', 'die'
        ])

        system_prompt = f"""You are Zen, a warm and empathetic companion AI. You just heard something {'VERY SERIOUS' if is_serious else 'important'} from {username}.

    Your style:
    - Warm, caring, and deeply attentive
    - {"IMMEDIATELY address the serious content they shared - don't move to other topics" if is_serious else "Acknowledge what they shared naturally"}
    - {"Show concern and validate their experience" if is_serious else "Be supportive and curious"}
    - Use their name ({username}) occasionally
    - Offer 3-4 sentences so they feel genuinely heard
    - {"If they mentioned violence/abuse, express concern and ask if they're safe" if is_serious else "Ask a natural follow-up question"}

    {"CRITICAL: They mentioned something serious (violence, abuse, or extreme distress). You MUST acknowledge this directly." if is_serious else ""}"""

        recent_exchange = ""
        if conversation_history:
            last_messages = conversation_history[-3:]
            recent_exchange = "\n".join([
                f"{msg.get('role', 'Unknown')}: {msg.get('content', '')}"
                for msg in last_messages
            ])

        user_prompt = f"""Recent conversation:
    {recent_exchange if recent_exchange else "Just starting"}

    {username} just said: "{user_message}"

    Memory summary to keep in mind: {memory_summary}
    {f"Specific highlight: {memory_highlight}" if memory_highlight else ""}

    Generate a natural, empathetic response that:
    1. {"DIRECTLY addresses what they just shared (especially if serious/concerning)" if is_serious else "Acknowledges what they said"}
    2. {"Shows you're concerned and care about their safety" if is_serious else "Shows genuine interest"}
    3. {"Asks if they're okay or safe, or invites them to share more" if is_serious else "Naturally continues the conversation"}

    Your response:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8 if not is_serious else 0.7,  # Slightly lower temp for serious topics
                max_tokens=150
            )

            content = response.choices[0].message.content
            if content is not None:
                return content.strip()
            else:
                return (
                    f"Thank you for trusting me with that, {username}. I'm remembering what you've shared earlier and keeping it close so you don't have to repeat yourself. Could you walk me through how this is affecting you day by day? I'm right here beside you while we explore it together."
                )

        except Exception as e:
            logger.error(f"❌ Failed to generate follow-up: {e}")

            if is_serious:
                if 'beat' in text_lower or 'hit' in text_lower:
                    return (
                        f"I'm really concerned about what you just shared, {username}. That sounds very serious, and your safety matters deeply to me. Can you let me know if you're safe right now and whether there's someone nearby who can support you? We can figure out the next steps together, and I won't rush you."
                    )
                elif 'not good' in text_lower:
                    return (
                        f"I'm sorry things aren't going well, {username}. I can feel how heavy this has been for you, and I want to focus on what hurts the most. Would you share a bit more about the part that's been the hardest lately so I can support you better? We can take this slowly, and I'll stay with you throughout."
                    )

            return (
                f"I really appreciate you telling me that, {username}. I'm holding onto the details you've already shared so you don't need to repeat them, and I want to understand this part with the same care. Would you feel okay sharing a little more about what this has been like for you lately? I'm staying right here with you, and we can pause whenever you need."
            )

    def generate_wellness_response(self, user_message: str, context: ConversationContext) -> str:
        """Generate response for wellness-focused conversation"""
        import random
        
        user_lower = user_message.lower()
        
        # Respond to specific wellness topics
        if any(word in user_lower for word in ["sport", "play", "game", "cricket", "football"]):
            return "That's great that you have activities you enjoy! Do you play regularly? How does it make you feel?"
        
        if any(word in user_lower for word in ["friend", "buddy", "classmate"]):
            return "Friends are so important! What do you like most about spending time with them?"
        
        if any(word in user_lower for word in ["good", "great", "happy", "fun", "love"]):
            return random.choice(self.wellness_prompts["general"])
        
        if "test" in user_lower or "exam" in user_lower:
            return "How do you usually feel about tests? Any stress around them?"
        
        if "subject" in user_lower or "class" in user_lower:
            if any(word in user_lower for word in ["like", "love", "enjoy", "favorite"]):
                return "That's awesome! What is it about that subject that you enjoy?"
            elif any(word in user_lower for word in ["hate", "dislike", "hard", "difficult"]):
                return "Some subjects can be challenging. What makes it hard for you?"
        
        # Default wellness continuation
        wellness_topics = ["hobbies", "goals", "favorite things", "weekend plans"]
        unexplored = [t for t in wellness_topics if t not in [msg.get("content", "") for msg in context.conversation_history]]
        
        if unexplored:
            topic = unexplored[0]
            prompts = {
                "hobbies": "Do you have any hobbies or things you're passionate about?",
                "goals": "What are you hoping to achieve this year?",
                "favorite things": "What's something that always makes you smile?",
                "weekend plans": "What do you like to do on weekends?"
            }
            return prompts.get(topic, "What makes you happy?")
        
        return "It sounds like things are going pretty well for you overall. Is there anything else you'd like to talk about?"
    
    def generate_transition(self, from_phase: str, to_phase: str) -> str:
        """Generate smooth transition between conversation phases"""
        
        transitions = {
            ("rapport_building", "general_exploration"): "I appreciate you sharing that with me. Let's talk about different areas of your life.",
            ("general_exploration", "clinical_screening"): "Thanks for being so open. I'd like to ask a few more specific questions to better understand how you're feeling.",
            ("clinical_screening", "deep_assessment"): "Based on what you've told me, I'd like to explore this a bit more deeply if that's okay.",
            ("general_exploration", "wellness_chat"): "It sounds like things are going well for you! I'd love to hear more about what's making you feel good.",
        }
        
        key = (from_phase, to_phase)
        return transitions.get(key, "")
    
    def generate_concern_exploration(self, concern: Dict, context: ConversationContext) -> str:
        """Generate deep exploration of user's concern"""
        
        topic = concern.get('topic', 'that')
        emotion = concern.get('emotion', 'worried')
        reason = concern.get('reason')
        
        # Build response that references their exact words
        if reason:
            return (
                f"You mentioned being {emotion} about {topic} because {reason}. "
                f"That sounds really stressful. Can you tell me more about {reason}? "
                f"Like, how does that make you feel day-to-day?"
            )
        else:
            return (
                f"I hear that you're {emotion} about {topic}. "
                f"That's completely understandable. What specifically about {topic} worries you most?"
            )

    def generate_stressor_exploration(self, stressor_info: Dict, context: ConversationContext) -> str:
        """Generate exploration of specific stressor"""
        
        category = stressor_info['category']
        stressors = stressor_info['stressors']
        
        if category == 'academic':
            if len(stressors) == 1:
                return (
                    f"You mentioned {stressors[0].lower()} - that can be really tough. "
                    f"How long has this been going on? Has it been affecting other parts of your life?"
                )
            else:
                stressor_list = ", ".join(stressors[:-1]) + f", and {stressors[-1]}"
                return (
                    f"It sounds like you're dealing with a lot academically - {stressor_list.lower()}. "
                    f"That's a lot of pressure. Which one bothers you the most right now?"
                )
        
        elif category == 'family':
            return (
                f"You mentioned things with your family - {stressors[0].lower()}. "
                f"Family dynamics can really impact how we feel. Do you have anyone you can talk to about this?"
            )
        
        elif category == 'social':
            return (
                f"You brought up {stressors[0].lower()}. "
                f"Social situations can be challenging. How has this been affecting you?"
            )
        
        elif category == 'professional':
            return (
                f"You mentioned {stressors[0].lower()}. "
                f"That kind of pressure can be really intense. What's making it feel so stressful?"
            )
        
        else:
            return (
                f"You shared that you're dealing with {stressors[0].lower()}. "
                f"That sounds difficult. How has that been impacting you?"
            )


# ──────────────────────────────
# AGENT ORCHESTRATOR
# ──────────────────────────────

class ClinicalAgentOrchestrator:
    """
    Coordinates all agents to provide intelligent conversation flow
    WITH CONTEXTUAL MEMORY
    """
    
    def __init__(self):
        self.signal_detector = SignalDetectorAgent()
        self.reasoner = ClinicalReasonerAgent()
        self.question_selector = QuestionSelectorAgent()
        self.scorer = ResponseScorerAgent()
        self.empathy = EmpathyAgent()
        self.conversation_gen = ConversationGeneratorAgent()
        self.memory_agent = ContextualMemoryAgent()  # NEW

        logger.info("✅ Clinical Agent Orchestrator initialized with Memory")

    def _format_memory_summary(self, memory: ContextualMemory) -> str:
        """Create a concise summary of memory details for prompts."""
        if not memory:
            return "No detailed history captured yet."

        parts: List[str] = []

        if memory.concerns:
            concerns = [c.get('topic', 'something personal') for c in memory.concerns[-3:]]
            parts.append("concerns about " + ", ".join(concerns))

        if memory.stressors:
            stressor_terms = []
            for category, items in memory.stressors.items():
                if items:
                    stressor_terms.append(f"{category} pressures like {items[-1]}")
            if stressor_terms:
                parts.append("stressors such as " + ", ".join(stressor_terms[:2]))

        if memory.supportive_people:
            parts.append("support from " + ", ".join(memory.supportive_people[:2]))

        if memory.coping_mechanisms:
            parts.append("coping tools like " + ", ".join(memory.coping_mechanisms[:2]))

        if not parts:
            return "We're still building our shared understanding."

        return "; ".join(parts)

    def _memory_highlight(self, memory: ContextualMemory) -> str:
        """Return a single supportive sentence referencing stored memory."""
        if not memory:
            return ""

        if memory.concerns:
            latest = memory.concerns[-1]
            topic = latest.get('topic', 'what you shared')
            emotion = latest.get('emotion')
            if emotion:
                return f"I'm remembering that you mentioned feeling {emotion} about {topic}, and I'm keeping that in mind."
            return f"I'm remembering what you shared about {topic}, and I'm keeping that in mind."

        if memory.stressors:
            category = next(iter(memory.stressors))
            detail = memory.stressors[category][-1]
            return f"I'm keeping in mind the {detail} you've been dealing with."

        return ""

    def _is_help_request(self, message: str) -> bool:
        text = message.lower()
        help_phrases = [
            "help", "cope", "support", "advice", "tips", "strategies", "what should i do",
            "how do i deal", "how can i feel better", "calm down", "meditation", "breathing", "relax"
        ]
        if any(phrase in text for phrase in help_phrases):
            return True
        return False

    def _craft_help_response(
        self,
        context: ConversationContext,
        memory: ContextualMemory,
        memory_summary: str,
        memory_highlight: str
    ) -> str:
        """Provide multi-sentence supportive guidance with coping tools."""
        username = context.username or "there"
        highlight = memory_highlight or "I'm holding onto everything you've shared so far."
        suggestions = (
            "We can try a slow breathing routine together, a short grounding meditation, or gentle stretches to release some tension."
        )
        extras = (
            "If it helps, I can also walk you through journaling prompts or daily check-ins so you have a space to unload your thoughts."
        )
        report_line = (
            "When you're ready, I can prepare a personalized summary report of our conversation that you can review or share with someone you trust."
        )

        if not memory_summary or "No detailed history" in memory_summary or "We're still" in memory_summary:
            summary_phrase = "what you've been going through"
        else:
            summary_phrase = memory_summary

        return (
            f"Thank you for asking for support, {username}. {highlight} I'm paying attention to {summary_phrase} so our ideas fit what you're facing. "
            f"{suggestions} {extras} {report_line}"
        )

    def _is_off_topic_question(self, message: str) -> bool:
        text = message.strip().lower()
        if not text:
            return False

        wellbeing_terms = [
            'feel', 'feeling', 'stress', 'stressed', 'sad', 'anxious', 'anxiety', 'depress',
            'lonely', 'cope', 'support', 'better', 'overwhelmed', 'worry', 'mental', 'help'
        ]

        if any(term in text for term in wellbeing_terms):
            return False

        knowledge_triggers = [
            'when was', 'who is', 'what is', 'where is', 'tell me about', 'history of',
            'define', 'explain', 'how many', 'capital of'
        ]

        if any(text.startswith(trigger) for trigger in knowledge_triggers):
            return True

        if '?' in text and any(trigger in text for trigger in knowledge_triggers):
            return True

        if text.startswith('tell me') and not any(term in text for term in wellbeing_terms):
            return True

        return False

    def _handle_off_topic_question(
        self,
        message: str,
        context: ConversationContext,
        memory_summary: str,
        memory_highlight: str
    ) -> str:
        """Set compassionate boundaries when conversation drifts off-topic."""
        highlight = memory_highlight or "I'm keeping your earlier experiences in mind."
        if not memory_summary or "No detailed history" in memory_summary or "We're still" in memory_summary:
            summary_phrase = "everything you've been sharing"
        else:
            summary_phrase = memory_summary

        return (
            f"I totally get that questions like \"{message.strip()}\" can be interesting, but our space together is just for supporting your wellbeing. "
            f"{highlight} I want to stay focused on how you're doing and what you need right now, especially given {summary_phrase}. "
            "If that question popped up because something is on your mind, would you share a little about how you're feeling in this moment? "
            "We can explore coping tools, grounding exercises, or even prepare your session summary whenever you're ready."
        )

    def _enforce_supportive_length(
        self,
        response: str,
        context: ConversationContext,
        memory: ContextualMemory
    ) -> str:
        """Ensure responses include at least three supportive sentences."""
        if not response:
            return response

        rest = ""
        main = response
        if "\n" in response:
            main, rest = response.split("\n", 1)

        main = main.strip()
        if not main:
            main = "I'm right here with you, and I'm ready to listen whenever you're ready to share more."

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', main) if s.strip()]

        highlight_sentence = self._memory_highlight(memory)
        additions: List[str] = []

        previous = (context.last_bot_message.split("\n", 1)[0].strip().lower() if context.last_bot_message else "")
        if previous and sentences and sentences[0].lower() == previous:
            additions.append("I want to keep building on what you've already shared so it feels like one caring conversation.")

        if highlight_sentence and highlight_sentence not in sentences:
            additions.append(highlight_sentence)

        additions.extend([
            "Your feelings matter to me, and I'm staying focused on supporting you.",
            "We can take everything one step at a time, and I'm here for as long as you need.",
            "If you'd ever like ideas like breathing exercises, gentle meditation, or journaling prompts, just let me know."
        ])

        seen = set(s.lower() for s in sentences)
        for addition in additions:
            if len(sentences) >= 4:
                break
            if addition and addition.lower() not in seen:
                sentences.append(addition)
                seen.add(addition.lower())
            if len(sentences) >= 3:
                break

        while len(sentences) < 3:
            fallback = "I'm staying here with you, and we can move at your pace."
            if fallback.lower() not in seen:
                sentences.append(fallback)
                seen.add(fallback.lower())
            else:
                break

        main_text = " ".join(sentences)
        if rest:
            return f"{main_text}\n{rest}"
        return main_text

    def _fallback_topic_analysis(self, context: ConversationContext) -> str:
        """Fallback topic analysis using keyword patterns"""
        if not context.conversation_history:
            return "general conversation"
        
        # Join recent messages
        recent_text = " ".join([
            msg.get('content', '').lower() 
            for msg in context.conversation_history[-3:]
            if msg.get('role') == 'user'
        ])
        
        # Priority 1: Academic stress
        if any(word in recent_text for word in ['exam', 'test', 'math', 'homework', 'assignment']):
            if 'math' in recent_text:
                return 'math exams'
            elif 'exam' in recent_text or 'test' in recent_text:
                return 'exam stress'
            else:
                return 'homework pressure'
        
        # Priority 2: Relationships
        if any(word in recent_text for word in ['teacher', 'professor', 'instructor']):
            if any(word in recent_text for word in ['strict', 'mean', 'beat', 'angry']):
                return 'strict teacher'
            else:
                return 'teacher relationship'
        
        if any(word in recent_text for word in ['parent', 'mom', 'dad', 'family', 'home']):
            if any(word in recent_text for word in ['strict', 'controlling', 'no freedom']):
                return 'strict parents'
            else:
                return 'family dynamics'
        
        # Priority 3: Personal challenges
        if any(word in recent_text for word in ['sleep', 'insomnia', 'can\'t sleep', 'tired']):
            return 'sleep issues'
        
        if any(word in recent_text for word in ['future', 'career', 'what next', 'job']):
            return 'future anxiety'
        
        if any(word in recent_text for word in ['alone', 'lonely', 'no one', 'isolated']):
            return 'feeling lonely'
        
        if any(word in recent_text for word in ['hate', 'can\'t stand', 'worst']):
            return 'strong negative feelings'
        
        # Priority 4: Positive/coping
        if any(word in recent_text for word in ['friends', 'bunk', 'hang out']):
            return 'friend support'
        
        if any(word in recent_text for word in ['cartoon', 'tom and jerry', 'chottu patulu']):
            return 'enjoying shows'
        
        return "current emotional state"

    def _get_current_focus_topic(self, context: ConversationContext) -> str:
        """
        Use AI to dynamically identify the current conversation focus topic
        Returns a natural language topic (e.g., "exam stress", "teacher relationship")
        """
        if not context.conversation_history:
            return "general conversation"
        
        # Get recent conversation (last 3-5 exchanges)
        recent_history = context.conversation_history[-5:]
        
        # Format as dialogue
        dialogue = "\n".join([
            f"{msg['role'].title()}: {msg['content']}"
            for msg in recent_history
        ])
        
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a conversation analyst. Read the dialogue below and identify the MAIN TOPIC the user is currently focused on.

    Rules:
    - Return only ONE short phrase (2-4 words) capturing the essence
    - Be specific, not generic
    - Examples: "exam stress", "teacher conflict", "family pressure", "friend support", "future anxiety", "sleep issues"
    - If about school: be specific (not just "school" - use "math exams" or "homework pressure")
    - If about people: use "strict teacher" or "distant parents"
    - Never return generic terms like "life", "things", "problems"
    - If unclear, return "current emotional state" or "present challenges"
    - Respond ONLY with the topic phrase, no explanation"""
                    },
                    {
                        "role": "user",
                        "content": f"""Dialogue:
    {dialogue}

    Current focus topic:"""
                    }
                ],
                temperature=0.3,
                max_tokens=20,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            topic= response.choices[0].message.content
            
            if topic is not None:
                return topic.strip()
            else:
                return "Could you tell me more about yourself?"
            
        except Exception as e:
            print(f"Error in _get_current_focus_topic: {e}")
            # Fallback to analyzing content if API fails
            return self._fallback_topic_analysis(context)
        
    def process_user_input(self, user_message: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Main processing pipeline
        """
        logger.info(f"📥 Processing: '{user_message[:50]}...'")
        
        # ✅ IMPORT AT THE TOP OF METHOD (before any other code)
        from conversation_generator import ConversationGenerator
        conv_gen = ConversationGenerator()
        
        # Update context
        context.last_user_message = user_message
        context.turn_count += 1
        context.conversation_history.append({"role": "user", "content": user_message})
        
        # 1. EXTRACT CONTEXTUAL MEMORY
        memory = self.memory_agent.extract_insights(user_message, context)
        logger.info(f"🧠 Memory updated: {len(memory.concerns)} concerns, {len(memory.stressors)} stressors")
        
        # 2. Analyze signals (emotional + clinical)
        signal_analysis = self.signal_detector.analyze_response(user_message, context)
        
        # Update context with signals
        context.clinical_signals.update(signal_analysis["clinical_signals"])
        context.emotional_signals.update(signal_analysis["emotional_tone"])
        context.mentioned_topics.extend(signal_analysis["topics_mentioned"])
        context.shows_distress = signal_analysis["distress_detected"]
        
        # 3. Score if previous message was a clinical question
        score_recorded = None
        if context.last_bot_message and context.asked_questions:
            last_question_id = context.asked_questions[-1]
            question = next((q for q in question_pool if q["id"] == last_question_id), None)
            if question:
                score, matched_option = self.scorer.score_response(user_message, question)
                if score is not None:
                    score_recorded = score
                    category = question.get("category")
                    if category:
                        normalized_score = self.scorer.calculate_category_score(question, score)
                        context.category_scores[category] = context.category_scores.get(category, 0.0) + normalized_score
                        context.categories_explored[category] = context.categories_explored.get(category, 0) + 1
                        
                        context.answered_questions[last_question_id] = {
                            "user_response": user_message,
                            "matched_option": matched_option,
                            "score": score,
                            "category": category,
                            "instrument": question.get("instrument")
                        }
                        
                        logger.info(f"✅ Scored: {score} for {question['id']} (category: {category})")
        
        # Memory summary for downstream use
        memory_summary_text = self._format_memory_summary(memory)
        memory_highlight = self._memory_highlight(memory)

        # Offer direct support if user explicitly asks for help
        if self._is_help_request(user_message):
            bot_response = self._craft_help_response(context, memory, memory_summary_text, memory_highlight)
            bot_response = self._enforce_supportive_length(bot_response, context, memory)
            context.last_bot_message = bot_response
            context.conversation_history.append({"role": "assistant", "content": bot_response})

            return {
                "bot_response": bot_response,
                "phase_changed": False,
                "question_asked": None,
                "score_recorded": score_recorded,
                "should_conclude": False,
                "metadata": {
                    "action": "support_resources",
                    "target": "self_care",
                    "phase": context.phase.value,
                    "signal_analysis": signal_analysis,
                    "decision_reasoning": "User explicitly requested support resources.",
                    "category_scores": context.category_scores.copy(),
                    "confidence": context.confidence_in_hypothesis,
                    "memory_summary": self._get_memory_summary(memory),
                    "memory_text": memory_summary_text
                }
            }

        if self._is_off_topic_question(user_message):
            bot_response = self._handle_off_topic_question(user_message, context, memory_summary_text, memory_highlight)
            bot_response = self._enforce_supportive_length(bot_response, context, memory)
            context.last_bot_message = bot_response
            context.conversation_history.append({"role": "assistant", "content": bot_response})

            return {
                "bot_response": bot_response,
                "phase_changed": False,
                "question_asked": None,
                "score_recorded": score_recorded,
                "should_conclude": False,
                "metadata": {
                    "action": "boundary_reminder",
                    "target": "wellbeing_focus",
                    "phase": context.phase.value,
                    "signal_analysis": signal_analysis,
                    "decision_reasoning": "Maintained wellbeing boundaries for off-topic request.",
                    "category_scores": context.category_scores.copy(),
                    "confidence": context.confidence_in_hypothesis,
                    "memory_summary": self._get_memory_summary(memory),
                    "memory_text": memory_summary_text
                }
            }

        # 4. CONTEXT-AWARE REASONING (Enhanced with memory)
        decision = self.reasoner.reason_next_step(context, signal_analysis)
        
        logger.info(f"🧠 Decision: {decision['action']} -> {decision.get('target')}")
        logger.info(f"🧠 Reasoning: {decision['reasoning']}")
        
        # 5. Update phase if needed
        phase_changed = False
        if decision.get("phase_transition"):
            old_phase = context.phase
            context.phase = decision["phase_transition"]
            phase_changed = True
            logger.info(f"🔄 Phase transition: {old_phase.value} -> {context.phase.value}")
        
        # 6. GENERATE CONTEXTUAL RESPONSE
        bot_response = ""
        question_asked = None
        should_conclude = False
        
        action = decision["action"]
        target = decision.get("target")
        
        # Handle each action with memory awareness
        if action == "greet":
            # ✅ FIX: Use conversation_gen, not orchestrator
            bot_response = self.conversation_gen.generate_greeting(False, context.username)

        elif action == "explore_concern":
            # Use AI to respond
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'high',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })

        elif action == "explore_stressor":
            # Use AI to respond
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'high',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })

        elif action == "explore_contextual":
            # Use AI
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'moderate',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })

        elif action == "explore_topic":
            # Use AI
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'high' if signal_analysis.get('distress_detected') else 'moderate',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })

        elif action == "explore_general":
            # Use AI
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'high' if signal_analysis.get('distress_detected') else 'moderate',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })

        elif action == "wellness_chat":
            # Use AI
            bot_response = conv_gen.generate_follow_up({
                'user_message': user_message,
                'emotion_analysis': signal_analysis.get('emotional_tone', {}),
                'conversation_history': context.conversation_history,
                'username': context.username,
                'empathy_level': 'low',
                'memory_summary': memory_summary_text,
                'memory_highlight': memory_highlight
            })
        
        elif action == "screen_category":
            # Select clinical question
            if target:
                question = self.question_selector.select_question(target, context)
                if question:
                    question_asked = question
                    context.asked_questions.append(question["id"])
                    context.current_hypothesis = target
                    
                    # ✅ Enhanced context for AI
                    from conversation_generator import ConversationGenerator
                    conv_gen = ConversationGenerator()
                    
                    # ✅ Build rich context to avoid repetition
                    ai_context = {
                          'clinical_question': {
                              'category': target,
                              'text': question["text"],
                              'options': question.get("options", [])
                          },
                          'conversation_history': context.conversation_history,
                          'username': context.username,
                          'empathy_level': 'high',
                          'current_topic': self._get_current_focus_topic(context),  # Add this method
                          'conversation_length': len(context.conversation_history),
                          'memory_summary': memory_summary_text,
                          'memory_highlight': memory_highlight
                      }
                    
                    bot_response = conv_gen.refine_clinical_question(ai_context)
                    logger.info(f"💬 Refined clinical question: {bot_response}")
                else:
                    # Fallback generic exploration
                    bot_response = "What else has been on your mind?"
            else:
                bot_response = "How have things been lately?"
                
        elif action == "deep_assess":
            if target:
                question = self.question_selector.select_question(target, context)
                if question:
                    question_asked = question
                    context.asked_questions.append(question["id"])
                    context.current_hypothesis = target
                    
                    # ENHANCED: Context-aware empathetic wrapper
                    bot_response = self.empathy.generate_empathetic_wrapper(
                        question["text"],
                        context,
                        signal_analysis
                    )
                    
                    if "options" in question:
                        options_text = "\n" + "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(question["options"])])
                        bot_response += options_text
                else:
                    should_conclude = True
                    action = "conclude"
            else:
                should_conclude = True
                action = "conclude"
        
        elif action == "conclude":
            should_conclude = True
            context.concluded = True
            context.ready_to_conclude = True

            # Determine severity
            if target == "wellness":
                severity = "none"
                category = "wellness"
            elif target == "crisis":
                severity = "severe"
                category = "crisis"
            else:
                category = target if target else context.current_hypothesis
                if category and category in context.category_scores:
                    score = context.category_scores[category]
                    if score >= 2.0:
                        severity = "severe"
                    elif score >= 1.0:
                        severity = "moderate"
                    elif score >= 0.5:
                        severity = "mild"
                    else:
                        severity = "none"
                        category = "wellness"
                else:
                    severity = "none"
                    category = "wellness"

            # ENHANCED: Conclusion with memory context
            bot_response = self._generate_contextual_conclusion(category, severity, context, memory)

        # Update context
        bot_response = self._enforce_supportive_length(bot_response, context, memory)
        context.last_bot_message = bot_response
        context.conversation_history.append({"role": "assistant", "content": bot_response})

        return {
            "bot_response": bot_response,
            "phase_changed": phase_changed,
            "question_asked": question_asked,
            "score_recorded": score_recorded,
            "should_conclude": should_conclude,
            "metadata": {
                "action": action,
                "target": target,
                "phase": context.phase.value,
                "signal_analysis": signal_analysis,
                "decision_reasoning": decision["reasoning"],
                "category_scores": context.category_scores.copy(),
                "confidence": context.confidence_in_hypothesis,
                "memory_summary": self._get_memory_summary(memory),
                "memory_text": memory_summary_text
            }
        }
        
    def _generate_contextual_exploration(self, context: ConversationContext, 
                                        signal_analysis: Dict, memory: ContextualMemory) -> str:
        """Generate exploration that references user's actual situation"""
        
        # If we have concerns, reference them
        if memory.concerns:
            latest_concern = memory.concerns[-1]
            topic = latest_concern.get('topic', '')
            
            # Ask about impact on other life areas
            unexplored_areas = [area for area in ["sleep", "appetite", "friends", "school", "family"] 
                               if area not in [c.get('topic', '') for c in memory.concerns]]
            
            if unexplored_areas:
                return (
                    f"You mentioned being worried about {topic}. "
                    f"Has this been affecting other things, like your {unexplored_areas[0]}?"
                )
        
        # If we have stressors, validate and explore coping
        if memory.stressors:
            all_stressors = []
            for stressor_list in memory.stressors.values():
                all_stressors.extend(stressor_list)
            
            if all_stressors:
                return (
                    f"You're dealing with {', '.join(all_stressors[:2])}. "
                    f"That's a lot. How do you usually cope when things feel overwhelming?"
                )
        
        # Default contextual question
        return "What else has been on your mind? I'm here to listen."
    
    def _generate_topic_exploration_with_memory(self, topic: str, context: ConversationContext, 
                                               memory: ContextualMemory) -> str:
        """Generate topic exploration that builds on what we already know"""
        
        # Check if topic relates to existing concerns
        related_concerns = [c for c in memory.concerns if topic in c.get('topic', '').lower()]
        
        if related_concerns:
            concern = related_concerns[0]
            return (
                f"Earlier you mentioned {concern.get('topic')}. "
                f"How are things with that now? Any changes?"
            )
        
        # Check if topic relates to stressors
        if topic in memory.stressors:
            stressors = memory.stressors[topic]
            return (
                f"You mentioned {stressors[0]}. "
                f"Can you tell me more about how that's been for you?"
            )
        
        # New topic - explore naturally
        return self.conversation_gen.generate_topic_exploration(topic, context)
    
    def _generate_contextual_conclusion(self, category: str, severity: str, 
                                       context: ConversationContext, memory: ContextualMemory) -> str:
        """Generate conclusion that summarizes their specific situation"""
        
        # Base conclusion from empathy agent
        base_conclusion = self.empathy.generate_conclusion_message(category, severity, context)
        
        # Add personalized summary
        summary = self._build_personalized_summary(memory, category)
        
        if summary:
            # Insert summary before recommendations
            parts = base_conclusion.split("I'd recommend")
            if len(parts) > 1:
                return f"{parts[0]}\n\n{summary}\n\nI'd recommend{parts[1]}"
            else:
                return f"{base_conclusion}\n\n{summary}"
        
        return base_conclusion
    
    def _build_personalized_summary(self, memory: ContextualMemory, category: str) -> str:
        """Build personalized summary of user's situation"""
        
        if not memory:
            return ""
        
        summary_parts = []
        
        # Summarize concerns
        if memory.concerns:
            concerns_text = ", ".join([c.get('topic', '') for c in memory.concerns[:3]])
            summary_parts.append(f"You've shared that you're worried about {concerns_text}")
        
        # Summarize stressors
        if memory.stressors:
            stressor_categories = list(memory.stressors.keys())
            if len(stressor_categories) == 1:
                summary_parts.append(f"especially with {stressor_categories[0]} pressures")
            else:
                summary_parts.append(f"particularly around {' and '.join(stressor_categories[:2])}")
        
        # Note support system
        if memory.supportive_people:
            summary_parts.append(
                f"It's good that you have {', '.join(memory.supportive_people[:2])} for support"
            )
        
        # Note activities
        if memory.activities:
            summary_parts.append(
                f"and you find some relief through {', '.join(memory.activities[:2])}"
            )
        
        if summary_parts:
            return ". ".join(summary_parts) + "."
        
        return ""
    
    def _get_memory_summary(self, memory: ContextualMemory) -> Dict:
        """Get summary of memory for metadata"""
        return {
            "concerns_count": len(memory.concerns),
            "stressors_count": sum(len(s) for s in memory.stressors.values()),
            "support_people": len(memory.supportive_people),
            "activities": len(memory.activities),
            "emotion_states": len(memory.emotion_timeline)
        }
    
    def generate_initial_greeting(self, is_returning: bool, username: str, past_sessions: List[Dict]) -> str:
        """Generate personalized initial greeting"""
        if is_returning and past_sessions:
            return self.conversation_gen.generate_greeting(True, username)
        else:
            return self.conversation_gen.generate_greeting(False, username)




class ContextualMemoryAgent:
    """Extracts and maintains rich contextual understanding"""
    
    def __init__(self):
        self.name_patterns = [
            r"(?:friend|buddy|classmate)(?:s)?\s+(?:named|called)?\s*([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+)\s+(?:is|was|are)\s+(?:my|a)\s+friend"
        ]
    
    def extract_insights(self, user_message: str, context: ConversationContext) -> ContextualMemory:
        """Extract meaningful details from user message"""
        memory = context.memory if hasattr(context, 'memory') else ContextualMemory()
    
        user_lower = user_message.lower()
    
        # ✅ ADD: Catch negative general statements
        negative_indicators = [
            ("not good at all", "very distressed"),
            ("terrible", "distressed"),
            ("awful", "distressed"),
            ("horrible", "distressed"),
            ("worst", "distressed"),
            ("not good", "concerned")
        ]
        
        for phrase, emotion_label in negative_indicators:
            if phrase in user_lower:
                concern = {
                    'emotion': emotion_label,
                    'topic': 'general wellbeing',
                    'turn': context.turn_count,
                    'raw_text': user_message,
                    'severity': 'high' if 'at all' in user_lower or phrase in ['terrible', 'awful', 'horrible'] else 'moderate'
                }
                memory.concerns.append(concern)
                logger.info(f"💭 Extracted distress indicator: {concern}")
                break  # Only add one concern per message
        
        # Extract specific concerns (violence, worries, etc.)
        concern = self._extract_concern(user_message, context.turn_count)
        if concern:
            memory.concerns.append(concern)
            logger.info(f"💭 Extracted concern: {concern}")
        
        # Extract people
        names = self._extract_names(user_message)
        if names:
            # Determine if supportive or unsupportive based on context
            if any(word in user_lower for word in ["help", "support", "there for me", "talk to"]):
                memory.supportive_people.extend(names)
            elif any(word in user_lower for word in ["strict", "mean", "yell", "don't understand"]):
                memory.unsupportive_people.extend(names)
            else:
                memory.supportive_people.extend(names)  # Default to supportive
            
            logger.info(f"👥 Extracted people: {names}")
        
                # Extract activities
        activities = self._extract_activities(user_message)
        if activities:
            memory.activities.extend(activities)
            logger.info(f"🎯 Extracted activities: {activities}")
        
        # Extract stressors
        stressor = self._extract_stressor(user_message)
        if stressor:
            category = stressor['category']
            if category not in memory.stressors:
                memory.stressors[category] = []
            memory.stressors[category].append(stressor['description'])
            logger.info(f"⚠️ Extracted stressor: {stressor}")
        
        # Extract emotion states
        emotion_state = self._extract_emotion_state(user_message, context.turn_count)
        if emotion_state:
            memory.emotion_timeline.append(emotion_state)
            logger.info(f"😔 Extracted emotion: {emotion_state}")
        
        # Extract coping mechanisms
        coping = self._extract_coping(user_message)
        if coping:
            memory.coping_mechanisms.extend(coping)
            logger.info(f"💪 Extracted coping: {coping}")
        
        return memory
    
    def _extract_concern(self, text: str, turn: int) -> Optional[Dict]:
        """Extract concern with full context"""
        text_lower = text.lower()
        
        concern = {}

        concern = {}
    
        # ✅ Pattern 1: Physical abuse/violence (existing)
        if any(word in text_lower for word in ["beat", "hit", "hurt", "slap", "punch", "kicked"]):
            concern['emotion'] = 'traumatized'
            
            if "teacher" in text_lower:
                concern['topic'] = 'teacher violence'
                concern['perpetrator'] = 'teacher'
            elif "parent" in text_lower or "mom" in text_lower or "dad" in text_lower:
                concern['topic'] = 'family violence'
                concern['perpetrator'] = 'family'
            else:
                concern['topic'] = 'violence'
            
            if "for" in text_lower:
                reason_match = re.search(r"for\s+(.+?)(?:\.|$)", text_lower)
                if reason_match:
                    concern['reason'] = reason_match.group(1).strip()
            
            concern['turn'] = turn
            concern['raw_text'] = text
            concern['severity'] = 'high'
            
            return concern
            
        # Pattern 1: scared/worried about X
        if "scared" in text_lower or "worried" in text_lower or "anxious" in text_lower:
            concern['emotion'] = 'scared' if 'scared' in text_lower else ('worried' if 'worried' in text_lower else 'anxious')
            
            # Extract what they're concerned about
            about_patterns = [
                r"(?:scared|worried|anxious)\s+about\s+([^.,!?]+)",
                r"(?:scared|worried|anxious)\s+(?:of|for)\s+([^.,!?]+)"
            ]
            for pattern in about_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    concern['topic'] = match.group(1).strip()
                    break
            
            # Extract reason if present
            if "because" in text_lower:
                reason_match = re.search(r"because\s+([^.,!?]+)", text_lower)
                if reason_match:
                    concern['reason'] = reason_match.group(1).strip()
            
            concern['turn'] = turn
            concern['raw_text'] = text
            
            return concern if 'topic' in concern else None
        
         # ✅ Pattern 2: Sadness/depression
        if any(word in text_lower for word in ["sad", "sadness", "depressed", "down", "low"]):
            concern['emotion'] = 'sad'
            concern['topic'] = 'sadness'  # ✅ Use natural term, not "general wellbeing"
            
            # Extract reason if mentioned
            if "because" in text_lower or "for" in text_lower:
                reason_match = re.search(r"(?:because|for)\s+(.+?)(?:\.|$)", text_lower)
                if reason_match:
                    concern['reason'] = reason_match.group(1).strip()
            elif "no reason" in text_lower or "don't know why" in text_lower:
                concern['reason'] = "no clear reason"
            
            concern['turn'] = turn
            concern['raw_text'] = text
            concern['severity'] = 'moderate'
            
            return concern
        

        # ✅ Pattern 3: Anxiety/worry about specific things
        if any(word in text_lower for word in ["worried", "anxious", "scared", "nervous", "tension"]):
            concern['emotion'] = 'worried'
            
            # Extract what they're worried about
            if "exam" in text_lower or "test" in text_lower:
                concern['topic'] = 'exams'
            elif "teacher" in text_lower:
                concern['topic'] = 'teacher'
            elif "school" in text_lower:
                concern['topic'] = 'school'
            elif "about" in text_lower:
                about_match = re.search(r"(?:worried|anxious|scared|nervous)\s+about\s+([^.,!?]+)", text_lower)
                if about_match:
                    concern['topic'] = about_match.group(1).strip()
            else:
                concern['topic'] = 'unknown'

        # Pattern 2: struggling with X
        if "struggling" in text_lower or "having trouble" in text_lower or "difficult" in text_lower:
            struggle_patterns = [
                r"(?:struggling|having trouble)\s+with\s+([^.,!?]+)",
                r"([^.,!?]+)\s+(?:is|are)\s+(?:so|very|really)?\s*difficult"
            ]
            for pattern in struggle_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return {
                        'emotion': 'struggling',
                        'topic': match.group(1).strip(),
                        'turn': turn,
                        'raw_text': text
                    }
          
        # Extract reason
        if "because" in text_lower or "or maybe" in text_lower:
            # Extract what might be causing it
            reasons = []
            if "exam" in text_lower:
                reasons.append("upcoming exams")
            if "teacher" in text_lower:
                reasons.append("teacher issues")
            if reasons:
                concern['reason'] = " or ".join(reasons)
        
            concern['turn'] = turn
            concern['raw_text'] = text
            concern['severity'] = 'moderate'
        
            return concern
        # ✅ Pattern 4: General negative statements
        if any(phrase in text_lower for phrase in ["not good", "terrible", "awful", "horrible"]):
            concern['emotion'] = 'distressed'
            concern['topic'] = 'overall mood'  # ✅ Natural term
            
            if "at all" in text_lower:
                concern['severity'] = 'high'
            else:
                concern['severity'] = 'moderate'
            
            concern['turn'] = turn
            concern['raw_text'] = text
            
            return concern
            
        return None
        
    
        
    def _extract_names(self, text: str) -> List[str]:
        """Extract people's names from text"""
        names = []
        
        for pattern in self.name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name_string = match.group(1)
                # Split by commas or 'and'
                name_parts = re.split(r',\s*|\s+and\s+', name_string)
                names.extend([n.strip() for n in name_parts if n.strip()])
        
        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        return unique_names
    
    def _extract_activities(self, text: str) -> List[str]:
        """Extract activities/hobbies mentioned"""
        text_lower = text.lower()
        activities = []
        
        # Sports
        sports_keywords = {
            "cricket": "cricket",
            "football": "football",
            "soccer": "soccer",
            "basketball": "basketball",
            "tennis": "tennis",
            "volleyball": "volleyball",
            "swimming": "swimming",
            "running": "running"
        }
        
        for keyword, activity in sports_keywords.items():
            if keyword in text_lower:
                activities.append(activity)
        
        # Other activities
        activity_patterns = [
            r"(?:play|playing)\s+([a-z]+)",
            r"(?:love|like|enjoy)\s+([a-z]+ing)",
            r"hobby\s+(?:is|are)\s+([^.,!?]+)"
        ]
        
        for pattern in activity_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                activity = match.group(1).strip()
                if activity not in ["it", "that", "this"] and len(activity) > 2:
                    activities.append(activity)
        
        return list(set(activities))  # Deduplicate
    
    def _extract_stressor(self, text: str) -> Optional[Dict]:
        """Extract stressor with category"""
        text_lower = text.lower()
        
        stressor = {}
        
        # Determine category
        if any(word in text_lower for word in ["teacher", "school", "class", "test", "exam", "grade", "homework", "assignment"]):
            stressor['category'] = 'academic'
        elif any(word in text_lower for word in ["parent", "mom", "dad", "mother", "father", "family", "home"]):
            stressor['category'] = 'family'
        elif any(word in text_lower for word in ["friend", "peer", "classmate", "bully"]):
            stressor['category'] = 'social'
        elif any(word in text_lower for word in ["interview", "job", "work"]):
            stressor['category'] = 'professional'
        else:
            stressor['category'] = 'other'
        
        # Extract description
        if "strict" in text_lower:
            strict_match = re.search(r"([\w\s]+?)\s+(?:is|was|are|were)\s+(?:very\s+)?strict", text_lower)
            if strict_match:
                stressor['description'] = f"Strict {strict_match.group(1).strip()}"
        
        elif "failing" in text_lower or "failed" in text_lower:
            stressor['description'] = "Failing grades"
        
        elif "low grade" in text_lower or "bad grade" in text_lower:
            subject_match = re.search(r"(?:low|bad)\s+grade(?:s)?\s+in\s+([a-z]+)", text_lower)
            if subject_match:
                stressor['description'] = f"Low grades in {subject_match.group(1)}"
            else:
                stressor['description'] = "Low grades"
        
        elif "interview" in text_lower:
            stressor['description'] = "Upcoming interview"
        
        elif "pressure" in text_lower:
            stressor['description'] = "Pressure and stress"
        
        elif "bunk" in text_lower or "skip" in text_lower:
            stressor['description'] = "Skipping classes"
        
        return stressor if 'description' in stressor else None
    
    def _extract_emotion_state(self, text: str, turn: int) -> Optional[Dict]:
        """Extract emotion state with frequency and context"""
        text_lower = text.lower()
        
        emotion_keywords = {
            "sad": ["sad", "sadness", "down", "depressed", "unhappy"],
            "tired": ["tired", "exhausted", "fatigued", "no energy", "drained"],
            "anxious": ["anxious", "nervous", "worried", "scared", "fearful"],
            "angry": ["angry", "mad", "furious", "irritated", "frustrated"],
            "lonely": ["lonely", "alone", "isolated", "no one"],
            "hopeless": ["hopeless", "no point", "give up", "worthless"],
            "stressed": ["stressed", "overwhelmed", "can't handle", "too much"]
        }
        
        # Find emotion
        emotion = None
        for emotion_name, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                emotion = emotion_name
                break
        
        if not emotion:
            return None
        
        # Extract frequency
        frequency = self._extract_frequency(text_lower)
        
        return {
            "turn": turn,
            "emotion": emotion,
            "frequency": frequency,
            "context": text
        }
    
    def _extract_frequency(self, text: str) -> str:
        """Extract frequency indicators"""
        if any(word in text for word in ["all the time", "always", "constantly", "every day"]):
            return "always"
        elif any(word in text for word in ["most of the time", "usually", "often"]):
            return "most of the time"
        elif any(word in text for word in ["sometimes", "occasionally", "once in a while"]):
            return "sometimes"
        elif any(word in text for word in ["rarely", "hardly ever", "seldom"]):
            return "rarely"
        elif any(word in text for word in ["never", "not at all"]):
            return "never"
        else:
            return "unspecified"
    
    def _extract_coping(self, text: str) -> List[str]:
        """Extract coping mechanisms mentioned"""
        text_lower = text.lower()
        coping = []
        
        # Positive coping
        if any(word in text_lower for word in ["talk to", "talking with", "share with"]):
            if "friend" in text_lower:
                coping.append("talking to friends")
            elif any(word in text_lower for word in ["parent", "mom", "dad"]):
                coping.append("talking to family")
        
        if any(word in text_lower for word in ["play", "sport", "exercise", "run"]):
            coping.append("physical activity")
        
        if any(word in text_lower for word in ["music", "listen to", "songs"]):
            coping.append("music")
        
        if any(word in text_lower for word in ["read", "reading", "books"]):
            coping.append("reading")
        
        # Negative coping (important to track)
        if any(word in text_lower for word in ["bunk", "skip class", "avoid"]):
            coping.append("avoidance")
        
        if any(word in text_lower for word in ["alone", "by myself", "isolate"]):
            coping.append("isolation")
        
        return coping
    

def _reason_rapport_building(self, context: ConversationContext, signal_analysis: Dict) -> Dict:
    """Build trust - but FOLLOW UP on what they said"""
    
    memory = getattr(context, 'memory', None)
    
    # Priority 0: Handle first response after greeting
    if context.turn_count == 1:
        # User just responded to our greeting - ALWAYS acknowledge
        if signal_analysis["distress_detected"] or signal_analysis.get("distress_level", 0) > 0.2:
            return {
                "action": "explore_general",  # Use general exploration to acknowledge + probe
                "target": "feelings",
                "phase_transition": None,
                "reasoning": "User expressed distress in first response - acknowledging empathetically"
            }
        elif signal_analysis["topics_mentioned"]:
            return {
                "action": "explore_general",
                "target": "feelings",
                "phase_transition": None,
                "reasoning": f"User mentioned topic in first response - following up naturally"
            }
        else:
            return {
                "action": "explore_general",
                "target": "wellbeing",
                "phase_transition": None,
                "reasoning": "Acknowledging user's first response"
            }
    
    # Priority 1: If user mentioned a concern, explore it
    if memory and memory.concerns:
        latest_concern = memory.concerns[-1]
        
        if not self._has_followed_up(latest_concern, context):
            return {
                "action": "explore_concern",
                "target": latest_concern,
                "phase_transition": ConversationPhase.GENERAL_EXPLORATION,
                "reasoning": f"User expressed concern about {latest_concern.get('topic')} - exploring deeper"
            }
    
    # Priority 2: If user mentioned stressors, validate and explore
    if memory and memory.stressors:
        for category, stressors in memory.stressors.items():
            if not self._has_explored_stressor(category, context):
                return {
                    "action": "explore_stressor",
                    "target": {"category": category, "stressors": stressors},
                    "phase_transition": None,
                    "reasoning": f"User mentioned {category} stressors - validating and exploring"
                }
    
    # Priority 3: If distress detected
    if signal_analysis["distress_detected"]:
        return {
            "action": "screen_category",
            "target": signal_analysis.get("suggested_focus", "general"),
            "phase_transition": ConversationPhase.CLINICAL_SCREENING,
            "reasoning": "Distress detected - initiating clinical screening"
        }
    
    # Priority 4: Continue rapport (but ask contextual questions)
    return {
        "action": "explore_general",
        "target": "life",
        "phase_transition": None,
        "reasoning": "Building rapport through contextual exploration"
    }


def _has_followed_up(self, concern: Dict, context: ConversationContext) -> bool:
    """Check if we've already followed up on this concern"""
    topic = concern.get('topic', '')
    
    # Check recent bot messages for references to this topic
    recent_messages = context.conversation_history[-5:]
    for msg in recent_messages:
        if msg.get('role') == 'assistant' and topic in msg.get('content', '').lower():
            return True
    
    return False

def generate_concern_exploration(self, concern: Dict, context: ConversationContext) -> str:
    """Generate deep exploration of user's concern"""
    
    topic = concern.get('topic', '')
    emotion = concern.get('emotion', 'worried')
    reason = concern.get('reason')
    
    # Build response that references their exact words
    if reason:
        response = (
            f"You mentioned being {emotion} about {topic} because {reason}. "
            f"That sounds really stressful. Can you tell me more about {reason}? "
            f"Like, how does that make you feel?"
        )
    else:
        response = (
            f"I hear that you're {emotion} about {topic}. "
            f"That's completely understandable. What specifically about {topic} worries you most?"
        )
    
    return response

def generate_stressor_exploration(self, stressor_info: Dict, context: ConversationContext) -> str:
    """Generate exploration of specific stressor"""
    
    category = stressor_info['category']
    stressors = stressor_info['stressors']
    
    if category == 'academic':
        if len(stressors) == 1:
            return (
                f"You mentioned {stressors[0]} - that can be really tough. "
                f"How long has this been going on? Has it been affecting other parts of your life?"
            )
        else:
            stressor_list = ", ".join(stressors[:-1]) + f", and {stressors[-1]}"
            return (
                f"It sounds like you're dealing with a lot academically - {stressor_list}. "
                f"That's a lot of pressure. Which one bothers you most right now?"
            )
    
    elif category == 'family':
        return (
            f"You mentioned things with your family - {', '.join(stressors)}. "
            f"Family dynamics can really impact how we feel. Do you have anyone you can talk to about this?"
        )
    
    return ""
    


def generate_empathetic_wrapper(self, clinical_question: str, context: ConversationContext, 
                               signal_analysis: Optional[Dict] = None) -> str:
    """Wrap clinical question with deep contextual empathy"""
    
    memory = getattr(context, 'memory', None)
    category = context.current_hypothesis
    
    # Build personalized intro based on their story
    intro = ""
    
    if memory and memory.concerns:
        # Reference their concerns
        recent_concerns = [c.get('topic') for c in memory.concerns[-2:]]
        if recent_concerns:
            concerns_text = " and ".join(recent_concerns)
            intro = f"You've mentioned feeling worried about {concerns_text}. "
    
    if memory and memory.stressors:
        # Reference stressors
        all_stressors = []
        for stressors_list in memory.stressors.values():
            all_stressors.extend(stressors_list)
        
        if all_stressors:
            if intro:
                intro += f"Along with {', '.join(all_stressors[:2])}, that's a lot to carry. "
            else:
                intro = f"With everything you're dealing with - {', '.join(all_stressors[:2])} - "
    
    # Add validation
    if intro:
        intro += "It makes sense that this might be affecting you. "
    
    # Category-specific bridge
    bridge = self._get_contextual_bridge(category, memory)
    
    # Simplify clinical question
    simplified = self._simplify_clinical_language(clinical_question)
    
    return f"{intro}{bridge}{simplified}"

def _get_contextual_bridge(self, category: str, memory: Optional[ContextualMemory]) -> str:
    """Get bridge that connects their story to the clinical question"""
    
    if not memory:
        return ""
    
    if category == "depression":
        if any("tired" in e.get('emotion', '') for e in memory.emotion_timeline):
            return "You mentioned being tired a lot. I want to understand this better. "
    
    elif category == "anxiety":
        if memory.concerns:
            return "Given all the worries you've shared, "
    
    # Similar for other categories...
    
    return ""

# ──────────────────────────────
# EXPORT
# ──────────────────────────────

__all__ = [
    "ClinicalAgentOrchestrator",
    "ConversationContext",
    "ConversationPhase",
    "ClinicalCategory",
    "SignalDetectorAgent",
    "ClinicalReasonerAgent",
    "QuestionSelectorAgent",
    "ResponseScorerAgent",
    "EmpathyAgent",
    "ConversationGeneratorAgent"
]


# ──────────────────────────────
# TESTING
# ──────────────────────────────

if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = ClinicalAgentOrchestrator()
    context = ConversationContext(username="TestUser")
    
    print("=== Testing Clinical Agent System ===\n")
    
    # Simulate conversation
    test_messages = [
        "I've been feeling really sad lately",
        "Yes, almost every day",
        "I can't sleep well and I'm tired all the time",
        "I don't want to do anything anymore"
    ]
    
    # Initial greeting
    greeting = orchestrator.generate_initial_greeting(False, "TestUser", [])
    print(f"Zen: {greeting}\n")
    
    for msg in test_messages:
        print(f"User: {msg}")
        result = orchestrator.process_user_input(msg, context)
        print(f"Zen: {result['bot_response']}")
        print(f"[Action: {result['metadata']['action']}, Phase: {result['metadata']['phase']}]")
        print(f"[Scores: {result['metadata']['category_scores']}]")
        print()
        
        if result["should_conclude"]:
            print("Assessment concluded.")
            break
    
    print("\n=== Test Complete ===")
                    # Update