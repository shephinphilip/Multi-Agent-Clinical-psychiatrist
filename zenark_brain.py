"""
Zenark Brain - Central Coordinator
Integrates with existing clinical agent orchestrator
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from zenark_db import ZenarkDB
from nrc_emotion_analyzer import NRCEmotionAnalyzer
from clinical_agents import ClinicalAgentOrchestrator, ConversationContext, ConversationPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZenarkBrain:
    """Central coordinator for Zenark multi-agent system"""
    
    def __init__(self, db_path: str = "zenark.db"):
        # Initialize components
        self.db = ZenarkDB(db_path)
        self.emotion_analyzer = NRCEmotionAnalyzer()
        self.orchestrator = ClinicalAgentOrchestrator()
        
        # Session state - properly typed
        self.current_user_id: Optional[int] = None
        self.current_session_id: Optional[str] = None
        self.current_context: Optional[ConversationContext] = None
        
        logger.info("🧠 Zenark Brain initialized successfully")
    
    def start_session(self, username: str) -> Dict:
        """Start new session for user"""
        # Get or create user
        user = self.db.get_user(username)
        
        if user:
            self.current_user_id = user['user_id']
            is_returning = True
            logger.info(f"👋 Welcome back: {username}")
            
            assert self.current_user_id is not None
            past_sessions = self.db.get_conversation_history(self.current_user_id, limit=10)
        else:
            self.current_user_id = self.db.create_user(username)
            is_returning = False
            past_sessions = []
            logger.info(f"✨ New user: {username}")
        
        # Create new session
        self.current_session_id = str(uuid.uuid4())
        
        # Initialize conversation context
        self.current_context = ConversationContext(
            username=username,
            phase=ConversationPhase.INITIAL_GREETING
        )
        
        # Generate greeting using orchestrator
        greeting = self.orchestrator.generate_initial_greeting(
            is_returning, 
            username, 
            past_sessions
        )
        
        # ✅ FIX: Immediately transition to RAPPORT_BUILDING after greeting
        self.current_context.phase = ConversationPhase.RAPPORT_BUILDING
        self.current_context.turn_count = 0  # Reset turn count for rapport phase
        
        # Update conversation history
        self.current_context.conversation_history.append({
            "role": "assistant",
            "content": greeting
        })
        
        # Type assertions
        assert self.current_user_id is not None
        assert self.current_session_id is not None
        
        # Save to database
        self.db.save_conversation(
            self.current_user_id,
            self.current_session_id,
            'Zen',
            greeting
        )
        
        return {
            'message': greeting,
            'is_returning': is_returning,
            'session_id': self.current_session_id
        }

    def process_user_input(self, user_input: str) -> Dict:
        """Main processing pipeline for user input"""
        if not self.current_user_id or not self.current_session_id or not self.current_context:
            return {'error': 'No active session'}
        
        # Update user activity
        self.db.update_user_activity(self.current_user_id)
        
        # Save user message to database
        self.db.save_conversation(
            self.current_user_id,
            self.current_session_id,
            'User',
            user_input
        )
        
        # Process through orchestrator
        try:
            result = self.orchestrator.process_user_input(user_input, self.current_context)
            
            bot_response = result['bot_response']
            should_conclude = result['should_conclude']
            metadata = result['metadata']
            
            # Save Zen's response
            self.db.save_conversation(
                self.current_user_id,
                self.current_session_id,
                'Zen',
                bot_response,
                emotion_signals=metadata.get('signal_analysis')
            )
            
            # If a clinical question was asked, save it
            if result.get('question_asked'):
                question = result['question_asked']
                if result.get('score_recorded') is not None:
                    # Find the previous question that was answered
                    if len(self.current_context.asked_questions) >= 2:
                        prev_question_id = self.current_context.asked_questions[-2]
                        answered_data = self.current_context.answered_questions.get(prev_question_id)
                        
                        if answered_data:
                            self.db.save_assessment_response(
                                self.current_user_id,
                                self.current_session_id,
                                prev_question_id,
                                self._get_question_text(prev_question_id),
                                answered_data.get('instrument', 'Unknown'),
                                answered_data.get('category', 'Unknown'),
                                answered_data['user_response'],
                                answered_data['score']
                            )
            
            # Update assessment state
            self._save_assessment_state(should_conclude, metadata)
            
            # Generate report if concluded
            report = None
            if should_conclude:
                report = self._generate_report(metadata)
            
            return {
                'message': bot_response,
                'phase': self.current_context.phase.value,
                'is_concluded': should_conclude,
                'report': report,
                'category_scores': self.current_context.category_scores,
                'questions_asked': len(self.current_context.asked_questions),
                'confidence': self.current_context.confidence_in_hypothesis,
                'is_crisis': metadata.get('action') == 'conclude' and metadata.get('target') == 'crisis'
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing input: {e}", exc_info=True)
            return {
                'error': f"An error occurred: {str(e)}",
                'message': "I'm sorry, I encountered an issue. Could you please repeat that?"
            }
    
    def _get_question_text(self, question_id: str) -> str:
        """Get question text from pool"""
        from question_pool import question_pool
        question = next((q for q in question_pool if q['id'] == question_id), None)
        return question['text'] if question else ""
    
    def _save_assessment_state(self, is_concluded: bool, metadata: Dict):
        """Save current assessment state to database"""
        # Guard against None values
        if not self.current_user_id or not self.current_session_id or not self.current_context:
            logger.warning("Cannot save assessment state - missing session data")
            return
        
        self.db.save_assessment_state(
            self.current_user_id,
            self.current_session_id,
            self.current_context.phase.value,
            self.current_context.category_scores,
            self._get_instruments_used(),
            len(self.current_context.asked_questions),
            self.current_context.confidence_in_hypothesis,
            is_concluded,
            metadata.get('decision_reasoning')
        )
    
    def _get_instruments_used(self) -> Dict[str, int]:
        """Get count of questions per instrument"""
        if not self.current_context:
            return {}
        
        instruments = {}
        for question_id, data in self.current_context.answered_questions.items():
            instrument = data.get('instrument', 'Unknown')
            instruments[instrument] = instruments.get(instrument, 0) + 1
        return instruments
    
    def _generate_report(self, metadata: Dict) -> Dict:
        """Generate assessment report"""
        # Guard against None values
        if not self.current_context:
            logger.error("Cannot generate report - no active context")
            return {}
        
        # Identify primary category and severity
        category_scores = self.current_context.category_scores
        
        if not category_scores or max(category_scores.values()) < 0.4:
            primary_category = 'general_wellness'
            severity = 'none'
        else:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
            score = category_scores[primary_category]
            
            if score >= 2.0:
                severity = 'severe'
            elif score >= 1.0:
                severity = 'moderate'
            elif score >= 0.5:
                severity = 'mild'
            else:
                severity = 'minimal'
        
        # Override if crisis detected
        if metadata.get('target') == 'crisis':
            primary_category = 'crisis'
            severity = 'severe'
        
        confidence = min(
            len(self.current_context.asked_questions) / 15.0,
            1.0
        )
        
        recommendations = self._generate_recommendations(primary_category, severity)
        
        # Save report to database (with None checks)
        if self.current_user_id and self.current_session_id:
            self.db.save_assessment_report(
                self.current_user_id,
                self.current_session_id,
                primary_category,
                severity,
                confidence,
                category_scores,
                self._get_instruments_used(),
                recommendations
            )
        
        report = {
            'primary_category': primary_category,
            'severity': severity,
            'confidence': confidence,
            'category_scores': category_scores,
            'instruments_used': self._get_instruments_used(),
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, category: str, severity: str) -> str:
        """Generate recommendations based on assessment"""
        recommendations = {
            'depression': {
                'minimal': 'Continue healthy habits. Stay connected with friends and family.',
                'mild': 'Consider talking to a school counselor or trusted adult. Practice self-care activities.',
                'moderate': 'Strongly recommend speaking with a mental health professional for support and guidance.',
                'severe': 'Please seek professional help urgently. Contact a mental health professional or crisis hotline.'
            },
            'anxiety': {
                'minimal': 'Keep up good stress management. Practice relaxation when needed.',
                'mild': 'Try relaxation techniques. Consider speaking with a counselor about managing worry.',
                'moderate': 'Recommend consulting a mental health professional for coping strategies.',
                'severe': 'Please seek professional help. Anxiety at this level benefits from professional treatment.'
            },
            'ptsd': {
                'minimal': 'Continue processing experiences in healthy ways.',
                'mild': 'Consider talking to a trauma-informed counselor.',
                'moderate': 'Strongly recommend working with a trauma specialist.',
                'severe': 'Please seek specialized trauma therapy urgently.'
            },
            'ocd': {
                'minimal': 'Continue healthy routines and patterns.',
                'mild': 'Consider speaking with a mental health professional about these patterns.',
                'moderate': 'Recommend consulting a therapist who specializes in OCD.',
                'severe': 'Please seek professional help from an OCD specialist.'
            },
            'bipolar': {
                'minimal': 'Continue monitoring mood patterns.',
                'mild': 'Important to discuss mood patterns with a mental health professional.',
                'moderate': 'Strongly recommend psychiatric evaluation.',
                'severe': 'Please seek psychiatric care urgently.'
            },
            'general_wellness': {
                'none': 'You\'re doing well! Continue healthy habits and reach out if things change.'
            },
            'crisis': {
                'severe': 'IMMEDIATE HELP NEEDED. Call 988 (Suicide & Crisis Lifeline) or 911. Tell a trusted adult now.'
            }
        }
        
        if category in recommendations and severity in recommendations[category]:
            return recommendations[category][severity]
        
        return 'Please consider speaking with a mental health professional for personalized guidance.'
    
    def force_conclude(self) -> Dict:
        """Force conclusion of assessment (user-initiated)"""
        logger.info("⏹️ User-initiated conclusion")
        
        # Guard against missing session data
        if not self.current_context:
            return {'error': 'No active session'}
        
        if not self.current_user_id or not self.current_session_id:
            return {'error': 'Session data incomplete'}
        
        if len(self.current_context.asked_questions) < 3:
            message = (
                f"I understand you'd like to wrap up, but we've only talked about a few things. "
                f"Would you like to continue for a bit longer so I can better understand how to help? "
                f"Or we can stop here if you prefer."
            )
            
            self.db.save_conversation(
                self.current_user_id,
                self.current_session_id,
                'Zen',
                message
            )
            
            return {
                'message': message,
                'phase': self.current_context.phase.value,
                'can_conclude': False,
                'is_concluded': False
            }
        
        # Generate conclusion
        category_scores = self.current_context.category_scores
        
        if not category_scores or max(category_scores.values()) < 0.4:
            category = 'wellness'
            severity = 'none'
        else:
            category = max(category_scores.items(), key=lambda x: x[1])[0]
            score = category_scores[category]
            severity = 'moderate' if score >= 1.0 else 'mild'
        
        from clinical_agents import EmpathyAgent
        empathy = EmpathyAgent()
        conclusion_message = empathy.generate_conclusion_message(
            category, 
            severity, 
            self.current_context
        )
        
        self.db.save_conversation(
            self.current_user_id,
            self.current_session_id,
            'Zen',
            conclusion_message
        )
        
        # Generate report
        report = self._generate_report({
            'action': 'conclude',
            'target': category
        })
        
        self.current_context.concluded = True
        
        return {
            'message': conclusion_message,
            'is_concluded': True,
            'can_conclude': True,
            'report': report,
            'primary_category': category,
            'severity': severity
        }
    
    def get_session_state(self) -> Dict:
        """Get current session state"""
        if not self.current_context:
            return {}
        
        return {
            'username': self.current_context.username,
            'phase': self.current_context.phase.value,
            'questions_asked': len(self.current_context.asked_questions),
            'category_scores': self.current_context.category_scores.copy(),
            'confidence_score': self.current_context.confidence_in_hypothesis,
            'is_concluded': self.current_context.concluded,
            'shows_distress': self.current_context.shows_distress
        }
    
    def get_conversation_history(self, limit: int = 50) -> List[Dict]:
        """Get conversation history for current session"""
        if not self.current_user_id or not self.current_session_id:
            return []
        
        return self.db.get_conversation_history(
            self.current_user_id,
            self.current_session_id,
            limit
        )
    
    def get_latest_report(self) -> Optional[Dict]:
        """Get latest assessment report"""
        if not self.current_user_id:
            return None
        
        return self.db.get_latest_report(self.current_user_id)