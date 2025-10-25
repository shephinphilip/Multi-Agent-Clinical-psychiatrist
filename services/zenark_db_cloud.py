"""
services/zenark_db_cloud.py
────────────────────────────
Loop-safe MongoDB management for Zenark (MongoDB Atlas + Motor).
"""

from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

# ──────────────────────────────
# INIT CONFIG
# ──────────────────────────────
load_dotenv()
logger = logging.getLogger("zenark.db")
logger.setLevel(logging.INFO)

MONGODB_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_DB", "zenark_db")

if not MONGODB_URI:
    raise ValueError("❌ Missing MONGO_URI in .env")

# ──────────────────────────────
# LOOP-SAFE DB INITIALIZATION
# ──────────────────────────────
# services/zenark_db_cloud.py

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_db() -> AsyncIOMotorDatabase:
    """Return a loop-safe MongoDB database handle."""
    global _client, _db
    current_loop = asyncio.get_event_loop()

    # Reconnect client if loop changed
    if _client is not None:
        try:
            if getattr(_client, '_loop', None) != current_loop:
                logger.warning("🔄 Detected loop change — reconnecting Motor client")
                _client.close()
                _client = None
                _db = None
        except Exception as e:
            logger.error(f"Error checking client loop: {e}")
            _client = None
            _db = None

    if _client is None:
        logger.info(f"🔁 Creating new Motor client for loop ID: {id(current_loop)}")
        _client = AsyncIOMotorClient(
            MONGODB_URI,
            maxIdleTimeMS=30000,
            connectTimeoutMS=20000,
            serverSelectionTimeoutMS=5000,
        )
        _db = _client[DB_NAME]

    if _db is None:
        raise RuntimeError("Failed to initialize database connection (_db is None)")

    return _db

# ──────────────────────────────
# 👤 USERS
# ──────────────────────────────
async def create_user(username: str, **kwargs) -> Optional[str]:
    db = get_db()
    users_col = db["users"]
    existing = await users_col.find_one({"username": username})
    if existing:
        return str(existing["_id"])
    user_doc = {
        "username": username,
        "age_band": kwargs.get("age_band"),
        "role": kwargs.get("role"),
        "language": kwargs.get("language"),
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
    }
    result = await users_col.insert_one(user_doc)
    return str(result.inserted_id)


async def get_user(username: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    users_col = db["users"]
    doc = await users_col.find_one({"username": username})
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    return None


async def update_user_activity(username: str) -> None:
    db = get_db()
    users_col = db["users"]
    await users_col.update_one(
        {"username": username},
        {"$set": {"last_active": datetime.utcnow()}}
    )


# ──────────────────────────────
# 💬 CONVERSATIONS
# ──────────────────────────────
async def save_conversation(username: str, session_id: str, speaker: str, message: str,
                            emotion_signals: Optional[Dict] = None) -> None:
    db = get_db()
    conv_col = db["conversations"]
    entry = {
        "username": username,
        "session_id": session_id,
        "speaker": speaker,
        "message": message,
        "emotion_signals": emotion_signals or {},
        "timestamp": datetime.utcnow(),
    }
    await conv_col.insert_one(entry)


async def get_conversation_history(username: str, session_id: Optional[str] = None,
                                   limit: int = 50) -> List[Dict]:
    db = get_db()
    conv_col = db["conversations"]
    query = {"username": username}
    if session_id:
        query["session_id"] = session_id
    cursor = conv_col.find(query).sort("timestamp", -1).limit(limit)
    return [
        {
            "speaker": doc["speaker"],
            "message": doc["message"],
            "timestamp": doc["timestamp"],
            "emotion_signals": doc.get("emotion_signals", {}),
        }
        async for doc in cursor
    ][::-1]


# ──────────────────────────────
# 🧠 AGENT INTERACTIONS
# ──────────────────────────────
async def save_interaction(user_id: str, user_input: str, response: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           agent: Optional[str] = None) -> None:
    db = get_db()
    interactions_col = db["agent_interactions"]
    doc = {
        "user_id": user_id,
        "agent": agent or "unknown_agent",
        "user_input": user_input,
        "response": response,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow(),
    }
    try:
        await interactions_col.insert_one(doc)
        logger.info(f"💾 Saved interaction for {agent or 'unknown'} (user={user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to save interaction: {e}")


# ──────────────────────────────
# 🧩 ASSESSMENT RESPONSES
# ──────────────────────────────
async def save_assessment_response(username: str, session_id: str, question_id: str,
                                   question_text: str, instrument: str, category: str,
                                   user_response: str, score: int) -> None:
    db = get_db()
    responses_col = db["assessment_responses"]
    entry = {
        "username": username,
        "session_id": session_id,
        "question_id": question_id,
        "question_text": question_text,
        "instrument": instrument,
        "category": category,
        "user_response": user_response,
        "score": score,
        "timestamp": datetime.utcnow(),
    }
    await responses_col.insert_one(entry)


# ──────────────────────────────
# 🧠 ASSESSMENT STATE
# ──────────────────────────────
async def save_assessment_state(username: str, session_id: str, phase: str,
                                category_scores: Dict, instruments_used: Dict,
                                questions_asked: int, confidence_score: float,
                                is_concluded: bool = False,
                                conclusion_reason: Optional[str] = None) -> None:
    db = get_db()
    state_col = db["assessment_state"]
    existing = await state_col.find_one({"username": username, "session_id": session_id})
    doc = {
        "username": username,
        "session_id": session_id,
        "phase": phase,
        "category_scores": category_scores,
        "instruments_used": instruments_used,
        "questions_asked": questions_asked,
        "confidence_score": confidence_score,
        "is_concluded": is_concluded,
        "conclusion_reason": conclusion_reason,
        "updated_at": datetime.utcnow(),
    }
    if existing:
        await state_col.update_one({"_id": existing["_id"]}, {"$set": doc})
    else:
        await state_col.insert_one(doc)


# ──────────────────────────────
# 📋 FINAL REPORTS
# ──────────────────────────────
async def save_assessment_report(username: str, session_id: str, primary_category: str,
                                 severity: str, confidence_score: float,
                                 category_scores: Dict, instruments_summary: Dict,
                                 recommendations: str) -> None:
    db = get_db()
    reports_col = db["assessment_reports"]
    entry = {
        "username": username,
        "session_id": session_id,
        "primary_category": primary_category,
        "severity": severity,
        "confidence_score": confidence_score,
        "category_scores": category_scores,
        "instruments_summary": instruments_summary,
        "recommendations": recommendations,
        "created_at": datetime.utcnow(),
    }
    await reports_col.insert_one(entry)


async def get_latest_report(username: str) -> Optional[Dict]:
    db = get_db()
    reports_col = db["assessment_reports"]
    doc = await reports_col.find_one({"username": username}, sort=[("created_at", -1)])
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    return None


# ──────────────────────────────
# 🗑️ DELETE USER
# ──────────────────────────────
async def delete_user(username: str) -> bool:
    db = get_db()
    users_col = db["users"]
    conv_col = db["conversations"]
    responses_col = db["assessment_responses"]
    state_col = db["assessment_state"]
    reports_col = db["assessment_reports"]
    interactions_col = db["agent_interactions"]

    result = await users_col.delete_one({"username": username})
    if result.deleted_count:
        await conv_col.delete_many({"username": username})
        await responses_col.delete_many({"username": username})
        await state_col.delete_many({"username": username})
        await reports_col.delete_many({"username": username})
        await interactions_col.delete_many({"user_id": username})
        logger.info(f"✅ Deleted all records for user: {username}")
        return True

    logger.warning(f"⚠️ User not found: {username}")
    return False
