import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Question:
    question_id: str
    question_text: str
    category: str
    instrument: str
    options: List[str]
    score_range: List[int]
    
class AdaptiveAssessmentEngine:
    """Manages adaptive clinical assessment logic"""
    
    def __init__(self, question_pool: List[Dict]):
        self.question_pool = question_pool
        self.categories = ['depression', 'anxiety', 'ptsd', 'ocd', 'bipolar']
        
        # Track which questions have been asked
        self.asked_questions = set()
        
        # Category exploration state
        self.category_exploration = {cat: [] for cat in self.categories}
        self.positive_categories = set()
        
        logger.info("✅ Adaptive Assessment Engine initialized")
    
    def get_initial_screening_questions(self) -> List[Question]:
        """Get first question from each category for initial screening"""
        screening_questions = []
        
        for category in self.categories:
            # Get first question from this category
            category_questions = [
                q for q in self.question_pool
                if q.get('category') == category and q.get('question_id') not in self.asked_questions
            ]
            
            if category_questions:
                question_data = category_questions[0]
                question = Question(
                    question_id=question_data['question_id'],
                    question_text=question_data['question_text'],
                    category=question_data['category'],
                    instrument=question_data['instrument'],
                    options=question_data['options'],
                    score_range=question_data['score_range']
                )
                screening_questions.append(question)
        
        return screening_questions
    
    def select_next_question(self, context: Dict) -> Optional[Question]:
        """Select next question based on adaptive algorithm"""
        phase = context.get('phase', 'exploration')
        category_scores = context.get('category_scores', {})
        questions_asked = context.get('questions_asked', 0)
        target_category = context.get('target_category')
        
        if phase == 'exploration':
            return self._select_exploration_question(category_scores)
        elif phase == 'deepening':
            return self._select_deepening_question(category_scores, target_category)
        elif phase == 'validation':
            return self._select_validation_question(category_scores, target_category)
        elif phase == 'casual':
            return None  # No structured questions in casual phase
        
        return None
    
    def _select_exploration_question(self, category_scores: Dict[str, float]) -> Optional[Question]:
        """Select question during exploration phase"""
        # Get categories that haven't been fully explored
        unexplored_categories = []
        
        for category in self.categories:
            asked_in_category = len(self.category_exploration[category])
            if asked_in_category < 2:  # Ask at least 2 questions per category
                unexplored_categories.append(category)
        
        if not unexplored_categories:
            return None
        
        # Prioritize categories with higher scores
        if category_scores:
            unexplored_categories.sort(
                key=lambda cat: category_scores.get(cat, 0.0),
                reverse=True
            )
        
        # Get next question from highest priority category
        for category in unexplored_categories:
            available_questions = [
                q for q in self.question_pool
                if q.get('category') == category and q.get('question_id') not in self.asked_questions
            ]
            
            if available_questions:
                question_data = available_questions[0]
                return Question(
                    question_id=question_data['question_id'],
                    question_text=question_data['question_text'],
                    category=question_data['category'],
                    instrument=question_data['instrument'],
                    options=question_data['options'],
                    score_range=question_data['score_range']
                )
        
        return None
    
    def _select_deepening_question(self, category_scores: Dict[str, float], 
                                   target_category: Optional[str] = None) -> Optional[Question]:
        """Select question during deepening phase"""
        # Identify positive categories (score >= 50% threshold)
        positive_cats = [
            cat for cat, score in category_scores.items()
            if score >= 0.5 and len(self.category_exploration.get(cat, [])) < 5
        ]
        
        if not positive_cats:
            return None
        
        # If target category specified, prioritize it
        if target_category and target_category in positive_cats:
            focus_category = target_category
        else:
            # Focus on highest scoring category
            positive_cats.sort(key=lambda cat: category_scores.get(cat, 0.0), reverse=True)
            focus_category = positive_cats[0]
        
        # Get next question from focus category
        available_questions = [
            q for q in self.question_pool
            if q.get('category') == focus_category and q.get('question_id') not in self.asked_questions
        ]
        
        if available_questions:
            question_data = available_questions[0]
            return Question(
                question_id=question_data['question_id'],
                question_text=question_data['question_text'],
                category=question_data['category'],
                instrument=question_data['instrument'],
                options=question_data['options'],
                score_range=question_data['score_range']
            )
        
        return None
    
    def _select_validation_question(self, category_scores: Dict[str, float], 
                                   target_category: Optional[str]) -> Optional[Question]:
        """Select validation question for identified category"""
        if not target_category:
            return None
        
        # Get remaining questions from target category
        available_questions = [
            q for q in self.question_pool
            if q.get('category') == target_category and q.get('question_id') not in self.asked_questions
        ]
        
        if available_questions:
            # Prefer questions from different instruments for validation
            used_instruments = set([
                q_id.split('_')[0] for q_id in self.category_exploration.get(target_category, [])
            ])
            
            diverse_questions = [
                q for q in available_questions
                if q.get('instrument') not in used_instruments
            ]
            
            question_data = diverse_questions[0] if diverse_questions else available_questions[0]
            
            return Question(
                question_id=question_data['question_id'],
                question_text=question_data['question_text'],
                category=question_data['category'],
                instrument=question_data['instrument'],
                options=question_data['options'],
                score_range=question_data['score_range']
            )
        
        return None
    
    def score_response(self, question: Question, user_response: str) -> Tuple[int, float]:
        """Score user's response to a question"""
        # Find matching option
        score = 0
        normalized_response = user_response.lower().strip()
        
        for i, option in enumerate(question.options):
            if normalized_response in option.lower():
                score = question.score_range[i]
                break
        
        # Calculate normalized score (0-1 scale)
        max_score = max(question.score_range) if question.score_range else 0
        normalized_score = score / max_score if max_score > 0 else 0.0
        
        return score, normalized_score
    
    def update_category_scores(self, category: str, normalized_score: float, 
                              current_scores: Dict[str, float]) -> Dict[str, float]:
        """Update category scores with new response"""
        if category not in current_scores:
            current_scores[category] = 0.0
        
        # Calculate weighted average based on number of questions asked
        questions_in_category = len(self.category_exploration.get(category, []))
        
        if questions_in_category == 0:
            current_scores[category] = normalized_score
        else:
            # Weighted moving average
            current_scores[category] = (
                (current_scores[category] * questions_in_category + normalized_score) /
                (questions_in_category + 1)
            )
        
        return current_scores
    
    def mark_question_asked(self, question_id: str, category: str):
        """Mark question as asked"""
        self.asked_questions.add(question_id)
        if category not in self.category_exploration:
            self.category_exploration[category] = []
        self.category_exploration[category].append(question_id)
    
    def determine_phase(self, category_scores: Dict[str, float], questions_asked: int) -> str:
        """Determine current assessment phase"""
        # Phase 1: Exploration (0-7 questions)
        if questions_asked < 8:
            # Check if all categories have been explored
            all_explored = all(
                len(self.category_exploration.get(cat, [])) >= 2 
                for cat in self.categories
            )
            return 'deepening' if all_explored and questions_asked >= 5 else 'exploration'
        
        # Phase 2: Deepening (8-12 questions)
        elif questions_asked < 13:
            # Check if we have a clear leading category
            if category_scores:
                max_score = max(category_scores.values()) if category_scores else 0.0
                if max_score >= 0.6:
                    return 'validation'
            return 'deepening'
        
        # Phase 3: Validation (12+ questions)
        else:
            return 'validation'
    
    def calculate_confidence(self, category_scores: Dict[str, float], questions_asked: int,
                           instruments_used: Dict[str, int]) -> float:
        """Calculate confidence in assessment"""
        confidence = 0.0
        
        # Factor 1: Number of questions (max 40%)
        question_factor = min(questions_asked / 15, 1.0) * 0.4
        
        # Factor 2: Score clarity (max 30%)
        if category_scores:
            sorted_scores = sorted(category_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                # Higher confidence if there's clear separation
                score_separation = sorted_scores[0] - sorted_scores[1]
                clarity_factor = min(score_separation * 2, 1.0) * 0.3
            else:
                clarity_factor = 0.15
        else:
            clarity_factor = 0.0
        
        # Factor 3: Instrument diversity (max 30%)
        num_instruments = len(instruments_used)
        diversity_factor = min(num_instruments / 3, 1.0) * 0.3
        
        confidence = question_factor + clarity_factor + diversity_factor
        
        return round(confidence, 2)
    
    def identify_primary_category(self, category_scores: Dict[str, float]) -> Tuple[str, str]:
        """Identify primary category and severity"""
        if not category_scores:
            return 'general_wellness', 'none'
        
        # Get highest scoring category - FIX: Proper type handling
        if len(category_scores) == 0:
            return 'general_wellness', 'none'
        
        primary_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        primary_score = category_scores[primary_category]
        
        # If all scores are low, user is likely well
        if primary_score < 0.4:
            return 'general_wellness', 'none'
        
        # Determine severity based on score
        if primary_score >= 0.8:
            severity = 'severe'
        elif primary_score >= 0.6:
            severity = 'moderate'
        elif primary_score >= 0.4:
            severity = 'mild'
        else:
            severity = 'minimal'
        
        return primary_category, severity
    
    def should_conclude_assessment(self, context: Dict) -> Tuple[bool, str]:
        """Determine if assessment should conclude"""
        category_scores = context.get('category_scores', {})
        questions_asked = context.get('questions_asked', 0)
        confidence = context.get('confidence_score', 0.0)
        
        # Reason 1: All categories show low scores (user is well)
        if category_scores and questions_asked >= 5:
            max_score = max(category_scores.values()) if category_scores else 0.0
            if max_score < 0.4:
                return True, 'all_categories_low'
        
        # Reason 2: High confidence reached
        if questions_asked >= 8 and confidence >= 0.75:
            return True, 'sufficient_confidence'
        
        # Reason 3: Maximum questions reached
        if questions_asked >= 15:
            return True, 'max_questions_reached'
        
        # Reason 4: Clear category identified with good data
        if questions_asked >= 10 and confidence >= 0.65:
            primary_category, severity = self.identify_primary_category(category_scores)
            if primary_category != 'general_wellness' and severity in ['moderate', 'severe']:
                return True, 'clear_category_identified'
        
        return False, ''
    
    def get_assessment_summary(self, context: Dict) -> Dict:
        """Generate assessment summary"""
        category_scores = context.get('category_scores', {})
        instruments_used = context.get('instruments_used', {})
        questions_asked = context.get('questions_asked', 0)
        confidence = context.get('confidence_score', 0.0)
        
        primary_category, severity = self.identify_primary_category(category_scores)
        
        return {
            'primary_category': primary_category,
            'severity': severity,
            'confidence': confidence,
            'category_scores': category_scores,
            'instruments_used': instruments_used,
            'questions_asked': questions_asked,
            'phase': context.get('phase', 'completed')
        }
    
    def reset_for_new_session(self):
        """Reset engine for new assessment session"""
        self.asked_questions.clear()
        self.category_exploration = {cat: [] for cat in self.categories}
        self.positive_categories.clear()
        logger.info("🔄 Assessment engine reset for new session")