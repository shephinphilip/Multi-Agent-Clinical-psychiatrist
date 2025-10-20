"""
Bridge module to load existing question pool in expected format
"""

from question_pool import question_pool as existing_pool
from typing import List, Dict

def load_question_pool() -> List[Dict]:
    """
    Convert existing question pool to format expected by AdaptiveAssessmentEngine
    
    Existing format:
    {
        "id": "phq9_1",
        "category": "depression",
        "instrument": "PHQ-9",
        "text": "Question text",
        "options": ["Option1", "Option2"],
        "score_range": [0, 1, 2]
    }
    
    Expected format (same):
    {
        "question_id": "phq9_1",
        "category": "depression",
        "instrument": "PHQ-9",
        "question_text": "Question text",
        "options": ["Option1", "Option2"],
        "score_range": [0, 1, 2]
    }
    """
    formatted_pool = []
    
    for question in existing_pool:
        formatted_question = {
            'question_id': question['id'],
            'question_text': question['text'],
            'category': question['category'],
            'instrument': question['instrument'],
            'options': question['options'],
            'score_range': question['score_range']
        }
        formatted_pool.append(formatted_question)
    
    return formatted_pool