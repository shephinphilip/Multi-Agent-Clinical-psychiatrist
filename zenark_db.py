# zenark_db.py
"""
Database management for Zenark
Handles all data persistence for users, conversations, and assessments
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZenarkDB:
    def __init__(self, db_path: str = "zenark.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    age_band TEXT,
                    role TEXT,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    emotion_signals TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Assessment responses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessment_responses (
                    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    instrument TEXT,
                    category TEXT,
                    user_response TEXT NOT NULL,
                    score INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Assessment state
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessment_state (
                    state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    category_scores TEXT,
                    instruments_used TEXT,
                    questions_asked INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 0.0,
                    is_concluded BOOLEAN DEFAULT 0,
                    conclusion_reason TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Final reports
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessment_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    primary_category TEXT,
                    severity TEXT,
                    confidence_score REAL,
                    category_scores TEXT,
                    instruments_summary TEXT,
                    recommendations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            conn.commit()
            logger.info("✅ Database initialized successfully")
    
    def create_user(self, username: str, age_band: Optional[str] = None, 
                    role: Optional[str] = None, language: Optional[str] = None) -> int:
        """Create or get user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO users (username, age_band, role, language)
                VALUES (?, ?, ?, ?)
            """, (username, age_band, role, language))
            
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            user_id = cursor.fetchone()[0]
            conn.commit()
            return user_id
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, age_band, role, language, 
                       created_at, last_active
                FROM users WHERE username = ?
            """, (username,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'username': row[1],
                    'age_band': row[2],
                    'role': row[3],
                    'language': row[4],
                    'created_at': row[5],
                    'last_active': row[6]
                }
        return None
    
    def update_user_activity(self, user_id: int):
        """Update last active timestamp"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_active = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
    
    def save_conversation(self, user_id: int, session_id: str, 
                         speaker: str, message: str, 
                         emotion_signals: Optional[Dict] = None):
        """Save conversation message"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (user_id, session_id, speaker, message, emotion_signals)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, session_id, speaker, message, 
                  json.dumps(emotion_signals) if emotion_signals else None))
            conn.commit()
    
    def get_conversation_history(self, user_id: int, 
                                session_id: Optional[str] = None,
                                limit: int = 50) -> List[Dict]:
        """Get conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT speaker, message, timestamp, emotion_signals
                    FROM conversations
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, session_id, limit))
            else:
                cursor.execute("""
                    SELECT speaker, message, timestamp, emotion_signals
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))
            
            rows = cursor.fetchall()
            return [{
                'speaker': row[0],
                'message': row[1],
                'timestamp': row[2],
                'emotion_signals': json.loads(row[3]) if row[3] else {}
            } for row in reversed(rows)]
    
    def save_assessment_response(self, user_id: int, session_id: str,
                                question_id: str, question_text: str,
                                instrument: str, category: str,
                                user_response: str, score: int):
        """Save assessment question response"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO assessment_responses
                (user_id, session_id, question_id, question_text,
                 instrument, category, user_response, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, question_id, question_text,
                  instrument, category, user_response, score))
            conn.commit()
    
    def save_assessment_state(self, user_id: int, session_id: str,
                            phase: str, category_scores: Dict,
                            instruments_used: Dict, questions_asked: int,
                            confidence_score: float, is_concluded: bool = False,
                            conclusion_reason: Optional[str] = None):
        """Save or update assessment state"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if state exists
            cursor.execute("""
                SELECT state_id FROM assessment_state
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("""
                    UPDATE assessment_state
                    SET phase = ?, category_scores = ?, instruments_used = ?,
                        questions_asked = ?, confidence_score = ?,
                        is_concluded = ?, conclusion_reason = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND session_id = ?
                """, (phase, json.dumps(category_scores), json.dumps(instruments_used),
                      questions_asked, confidence_score, is_concluded, conclusion_reason,
                      user_id, session_id))
            else:
                cursor.execute("""
                    INSERT INTO assessment_state
                    (user_id, session_id, phase, category_scores, instruments_used,
                     questions_asked, confidence_score, is_concluded, conclusion_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, session_id, phase, json.dumps(category_scores),
                      json.dumps(instruments_used), questions_asked, confidence_score,
                      is_concluded, conclusion_reason))
            
            conn.commit()
    
    def save_assessment_report(self, user_id: int, session_id: str,
                             primary_category: str, severity: str,
                             confidence_score: float, category_scores: Dict,
                             instruments_summary: Dict, recommendations: str):
        """Save final assessment report"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO assessment_reports
                (user_id, session_id, primary_category, severity,
                 confidence_score, category_scores, instruments_summary,
                 recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, primary_category, severity,
                  confidence_score, json.dumps(category_scores),
                  json.dumps(instruments_summary), recommendations))
            conn.commit()
    
    def get_latest_report(self, user_id: int) -> Optional[Dict]:
        """Get most recent assessment report"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT report_id, session_id, primary_category, severity,
                       confidence_score, category_scores, instruments_summary,
                       recommendations, created_at
                FROM assessment_reports
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'report_id': row[0],
                    'session_id': row[1],
                    'primary_category': row[2],
                    'severity': row[3],
                    'confidence_score': row[4],
                    'category_scores': json.loads(row[5]),
                    'instruments_summary': json.loads(row[6]),
                    'recommendations': row[7],
                    'created_at': row[8]
                }
        return None
    
    def delete_user(self, username: str) -> bool:
        """Delete user and all associated data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get user_id
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                
                # Delete all related data
                cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM assessment_responses WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM assessment_state WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM assessment_reports WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                
                conn.commit()
                logger.info(f"✅ User {username} deleted successfully")
                return True
        
        logger.warning(f"⚠️ User {username} not found")
        return False