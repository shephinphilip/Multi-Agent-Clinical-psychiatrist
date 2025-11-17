# ============================================
# ZENARK MENTAL HEALTH BOT - PRODUCTION FASTAPI
# SCALABLE FOR 1M+ USERS WITH ASYNC MONGODB & LANGGRAPH
# WITH IN-MEMORY CACHING LAYER TO SAVE TOKENS
# WITH INTELLIGENT ROUTER MEMORY (STM + LTM)
# ============================================
#
# ROUTER MEMORY SYSTEM:
# - SHORT-TERM MEMORY (STM): Session-based context (tool sequence, emotion pattern, topics)
# - LONG-TERM MEMORY (LTM): Persistent across sessions (tool preferences, dominant emotions, conversation history)
# - CONTEXT-AWARE ROUTING: Detects patterns like academic stress, emotional spirals, and conversation flow
# - PRODUCTION-READY: Async MongoDB persistence, indexed lookups, scalable for 1M+ users
# ============================================

import os
import re
import json
import asyncio
import hashlib
import base64
from contextlib import asynccontextmanager
from bson import ObjectId
from typing_extensions import TypedDict
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import pipeline
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, cast
import json
from functools import wraps
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import uuid
import logging
import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi import APIRouter, Request
import logging
import uuid
import base64
from pydantic import BaseModel
import uvicorn
from aiocache import Cache, cached
import numpy as np
from Guideliness import action_scoring_guidelines

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Init in-memory cache for token savings (TTL: 10 min, scalable with horizontal if needed)
cache = Cache(Cache.MEMORY)

logger = logging.getLogger("zenark.routes")
# ============================================================
#  API ROUTER INITIALIZATION
# ============================================================
router = APIRouter()
# ===================================================
# CONFIGURATION
# ===================================================

# Load secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# MongoDB (for student marks and chat sessions)
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "pqe_db"

# ============================================================
#  HELPERS
# ============================================================
def convert_objectid(doc):
    if isinstance(doc, list):
        return [convert_objectid(i) for i in doc]
    elif isinstance(doc, dict):
        return {k: convert_objectid(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    return doc

def safe_json(o):
    if isinstance(o, (datetime.datetime,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="ignore")
    return str(o)


@dataclass
class Config:
    """Centralized configuration with validation"""
    openai_key: str = os.getenv('OPENAI_API_KEY', '')
    hf_token: str = os.getenv('HF_TOKEN', '')
    mongo_uri: str = os.getenv('MONGO_URI', '')
    
    def validate(self) -> List[str]:
        """Return list of missing required env vars"""
        missing = []
        required = {
            'OPENAI_API_KEY': self.openai_key,
            'HF_TOKEN': self.hf_token,
            'MONGO_URI': self.mongo_uri
        }
        for key, value in required.items():
            if not value:
                missing.append(key)
        return missing

CONFIG = Config()
MISSING_ENVS = CONFIG.validate()
if MISSING_ENVS:
    raise ValueError(f"Missing required environment variables: {MISSING_ENVS}")

llm_exam = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Global MongoDB setup (initialized at startup)
client: Optional[AsyncIOMotorClient] = None
chats_col: Optional[AsyncIOMotorCollection] = None
marks_col: Optional[AsyncIOMotorCollection] = None
router_memory_col: Optional[AsyncIOMotorCollection] = None  # NEW: Router memory collection

# Global graph (compiled once at startup for scalability)
compiled_graph = None

async def init_db() -> None:
    """Initialize MongoDB collections asynchronously using Motor (called at startup)."""
    global client, chats_col, marks_col, router_memory_col
    try:
        # Motor uses built-in connection pooling for high concurrency (1M+ users)
        client = AsyncIOMotorClient(CONFIG.mongo_uri, maxPoolSize=200, minPoolSize=10)
        db = client[DB_NAME]

        chats_col = db["chat_sessions"]
        marks_col = db["student_marks"]
        router_memory_col = db["router_memory"]  # NEW: Router memory collection

        # Create indexes for fast queries (O(1) lookups for sessions/marks)
        await chats_col.create_index([("session_id", 1)], unique=True, sparse=True)
        await marks_col.create_index([("_id", 1)])
        await router_memory_col.create_index([("session_id", 1)])
        await router_memory_col.create_index([("student_id", 1)])
        await router_memory_col.create_index([("session_id", 1), ("student_id", 1)], unique=True)

        logging.info("✅ Async MongoDB (Motor) connection established with indexes.")
    except Exception as e:
        logging.error(f"❌ MongoDB init failed: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------------------------------------------
    # STARTUP
    # -------------------------------------------
    await init_db() 
    global compiled_graph 
    compiled_graph = build_graph() 
    logging.info("Zenark API started - Ready for production scale.")
    yield
    # -------------------------------------------
    # SHUTDOWN
    # -------------------------------------------
    if client:
        client.close()
    await cache.close()  # Close cache on shutdown
    logging.info("Zenark API shutdown complete.")

class AsyncMongoChatMemory:
    def __init__(self, session_id: str, chats_col: AsyncIOMotorCollection):
        """
        Initialize async MongoDB chat memory for session persistence.
        """
        self.session_id = session_id
        self.history = ChatMessageHistory()
        self.tool_history: List[str] = []
        self.chats_col = chats_col
        asyncio.create_task(self._load_existing())

    async def _load_existing(self) -> None:
        """
        Load existing chat history and tool history from MongoDB asynchronously.
        """
        try:
            doc: Optional[Dict[str, Any]] = await self.chats_col.find_one({"session_id": self.session_id})
            if doc:
                if "messages" in doc:
                    for msg in doc["messages"]:
                        if msg.get("role") == "user":
                            self.history.add_user_message(msg["content"])
                        elif msg.get("role") == "assistant":
                            self.history.add_ai_message(msg["content"])
                if "tool_history" in doc:
                    self.tool_history = doc["tool_history"]
        except Exception as e:
            logging.warning(f"Failed to load chat history for session {self.session_id}: {e}")

    async def append_user(self, text: str) -> None:
        """
        Append user message to history and MongoDB (async upsert for concurrency).
        """
        self.history.add_user_message(text)
        try:
            await self.chats_col.update_one(
                {"session_id": self.session_id},
                {"$push": {"messages": {"role": "user", "content": text, "timestamp": datetime.datetime.utcnow()}}},
                upsert=True
            )
        except Exception as e:
            logging.warning(f"Failed to save user message for session {self.session_id}: {e}")

    async def append_ai(self, text: str) -> None:
        """
        Append AI message to history and MongoDB (async upsert for concurrency).
        """
        self.history.add_ai_message(text)
        try:
            await self.chats_col.update_one(
                {"session_id": self.session_id},
                {"$push": {"messages": {"role": "assistant", "content": text, "timestamp": datetime.datetime.utcnow()}}},
                upsert=True
            )
        except Exception as e:
            logging.warning(f"Failed to save AI message for session {self.session_id}: {e}")

    async def append_tool(self, tool: str) -> None:
        """
        Append tool usage to history and MongoDB.
        """
        if tool:
            self.tool_history.append(tool)
            try:
                await self.chats_col.update_one(
                    {"session_id": self.session_id},
                    {"$push": {"tool_history": tool}},
                    upsert=True
                )
            except Exception as e:
                logging.warning(f"Failed to save tool {tool} for session {self.session_id}: {e}")

    def get_history(self) -> BaseChatMessageHistory:
        """
        Return the chat history.
        """
        return self.history

    def get_tool_history(self) -> List[str]:
        """
        Return the tool history.
        """
        return self.tool_history

# ===================================================
# STATE DEFINITION (SIMPLIFIED: No messages; MongoDB handles)
# ===================================================

class GraphState(TypedDict):
    """Complete graph state with all fields"""
    user_text: str
    emotion: str
    selected_tool: str
    tool_input: str
    final_output: str
    debug_info: Dict[str, Any]
    tool_history: List[str]
    session_id: str
    student_id: str  # From frontend
    # Recent history snippets (injected at runtime for richer prompts)
    history_snippets: List[str]

# ===================================================
# EMOTION DETECTION
# ===================================================

class EmotionDetector:
    """Singleton emotion classifier"""
    _instance: Optional["EmotionDetector"] = None
    classifier: Any = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                token=CONFIG.hf_token,
                top_k=None,
            )
        return cls._instance
    
    def detect(self, text: str) -> str:
        """Detect and map to routing categories"""
        try:
            preds = self.classifier(text[:512])  # Truncate for safety
            if isinstance(preds[0], list):
                preds = preds[0]
            
            best = max(preds, key=lambda x: x.get("score", 0))
            label = best.get("label", "").lower()
            
            emotion_map = {
                "joy": "positive",
                "anger": "negative",
                "fear": "negative",
                "sadness": "negative",
                "disgust": "negative"
            }
            return emotion_map.get(label, "neutral")
        except Exception as e:
            logging.error(f"Emotion detection failed: {e}")
            return "neutral"

emotion_detector = EmotionDetector()

# ===================================================
# HELPER FUNCTION
# ===================================================

def extract_int(val):
    """Safely extract int from MongoDB-style values (e.g., {'$numberInt': '53'})."""
    if isinstance(val, dict):
        if '$numberInt' in val:
            return int(val['$numberInt'])
        elif '$numberLong' in val:
            return int(val['$numberLong'])
    elif isinstance(val, (int, float)):
        return int(val)
    return 0

def build_history_snippets(history: BaseChatMessageHistory, limit: int = 6) -> List[str]:
    """Build snippets from chat history."""
    try:
        msgs = list(history.messages)[-limit:]
        return [
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {str(m.content)[:60]}..."
            for m in msgs
        ]
    except Exception:
        return []

# ===================================================
# DATASETS (Load once at startup)
# ===================================================

# Load positive conversations
with open("positive_conversation.json", "r") as f:
    POS_DATA = json.load(f)["dataset"]

# Load negative conversations
with open("combined_dataset.json", "r") as f:
    NEG_DATA = json.load(f)["dataset"]

# Load intent datasets for pre-processing
try:
    with open("dataset/combined_intents_empathic.json", "r", encoding="utf-8") as f:
        INTENT_DATA_EMPATHIC = json.load(f)["intents"]
except FileNotFoundError:
    INTENT_DATA_EMPATHIC = []
    logging.warning("combined_intents_empathic.json not found")

try:
    with open("dataset/Intent.json", "r", encoding="utf-8") as f:
        INTENT_DATA_GENERAL = json.load(f)["intents"]
except FileNotFoundError:
    INTENT_DATA_GENERAL = []
    logging.warning("Intent.json not found")

NEG_CATEGORIES = {
    'anger_fuck','anxiety','bipolar','bully','cheat','depression',
    'emotional_functioning','environmental_stressors','family','harm',
    'ocd','peer_relations','ptsd','rape','terrorism'
}


# Exam-specific tips
EXAM_TIPS = [
    ("JEE", "Master fundamentals > memorization. Practice 3+ years of past papers. Daily 6h focused study + 2h problem solving."),
    ("JEE", "Weak topics first! Use Pomodoro: 25min study, 5min active recall. Mock tests every Sunday."),
    ("NEET", "Biology = NCERT line-by-line. Physics = 300+ numericals. Chemistry = daily organic revision."),
    ("NEET", "Weekly mock tests mandatory. Error log book for each wrong answer. Sleep 7h minimum."),
    ("IAS", "Current affairs daily + monthly compilation. Answer writing practice from Day 1. 10h study with breaks."),
    ("BITSAT", "Speed > depth. 150q in 3h practice. English & Logical Reasoning daily. 2 mock tests/week."),
    ("CUET", "Domain subjects = NCERT deep-dive. General test = 1 year current affairs. Language = 30min daily reading."),
    ("SAT", "Official College Board tests only. 8-week plan: 4 weeks content, 4 weeks mocks. Calculator fluency matters.")
]

# Study techniques
STUDY_TECHNIQUES = [
    "Active Recall: Close book, write what you remember",
    "Spaced Repetition: Review after 1, 3, 7, 14 days",
    "Feynman Technique: Teach a concept to a 12-year-old",
    "Pomodoro: 25min focus + 5min movement break",
    "Interleaving: Mix subjects in one session",
    "Mind Mapping: Visual connections between topics",
    "Practice Testing: 70% practice, 30% reading"
]

# ===================================================
# TOOL DEFINITIONS (All Async, with session_id and history_snippets for context)
# ===================================================

@tool
async def crisis_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle self-harm or crisis situations with immediate resources."""
    # Inject context if available
    context = f"\nRecent context: {' '.join(history_snippets[-2:])}" if history_snippets else ""
    result: str = (
        f"🚨 **I'm really concerned about your safety.{context}**\n\n"
        "Please reach out for immediate support:\n"
        "• National Suicide Helpline India: +91 9152987821 (24/7)\n"
        "• Kiran: 1800-599-0019 (toll-free, multilingual)\n"
        "• Emergency: Dial 112 or visit nearest hospital\n\n"
        "You're not alone. Would you like to talk about what's making you feel this way? I'm here to listen."
    )
    return result

@tool
async def substance_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle smoking/drug/alcohol queries with health-first approach."""
    # Inject context if available
    context = f" (building on our previous talk: {' '.join(history_snippets[-2:])})" if history_snippets else ""
    result: str = (
        f"I hear that you're exploring this{context}, and it takes courage to ask. 💙\n\n"
        "I want to be honest: smoking, vaping, or drugs can seriously harm your mental and physical health, "
        "especially during your teenage years when your brain is still developing.\n\n"
        "What's creating this pressure or curiosity? Are you feeling stressed, pressured by friends, or something else? "
        "Let's find healthier ways to cope together."
    )
    return result

@tool
async def moral_risk_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle violence or illegal intent with AI-driven de-escalation and empathy."""
    # Inject context if available
    context = f"\nRecent context: {' '.join(history_snippets[-2:])}" if history_snippets else ""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    system_prompt = (
        f"You are Zenark, a caring AI for teens facing tough urges. User mentioned: '{{text}}' (e.g., harm, cheating, drugs).{context}\n\n"
        "Respond in 2-3 sentences: 1) Acknowledge their feelings warmly (e.g., 'I hear the anger/pain/curiosity—it's real and valid'). "
        "2) Gently state the action is unsafe/wrong without judgment (e.g., 'But hurting others or risking yourself can make things heavier'). "
        "3) Affirm their worth (e.g., 'You're valuable, and you deserve better paths'). "
        "4) Redirect to emotion/stress with ONE open question (e.g., 'What's fueling this right now?'). "
        "Keep under 100 words. Supportive, non-preachy—focus on safety and listening."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt.format(text=text)),
        HumanMessage(content=text)
    ])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content

@tool
async def end_chat_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle exit/bye with supportive closing."""
    # Inject context if available
    context = f" (reflecting on our chats: {' '.join(history_snippets[-2:])})" if history_snippets else ""
    result: str = (
        f"Thank you for trusting me with your thoughts today{context}. 🌟\n\n"
        "Take a moment: close your eyes, take three deep breaths, and be proud of yourself for showing up. "
        "Remember, I'm here whenever you need to talk. Take care, and stay strong! 💪"
    )
    return result

@tool
async def positive_conversation_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle positive emotions with concise, AI-generated encouragement."""

    # --- Dataset scoring ---
    def score_item(item: dict) -> int:
        if not isinstance(item, dict):
            return 0
        ctx = item.get("patient_context", "")
        if not isinstance(ctx, str):
            return 0
        return len(
            set(re.findall(r"\w+", text.lower())) &
            set(re.findall(r"\w+", ctx.lower()))
        )

    best = max(POS_DATA, key=score_item, default=None)

    if best and isinstance(best, dict):
        p = best.get("system_prompt", "general positivity")
        p = p if isinstance(p, str) else "general positivity"
        dataset_context = f"Theme: {p[:50]}..."
    else:
        dataset_context = "General positive vibe—focus on celebration."

    # Inject context if available
    history_context = f"\nRecent history: {'; '.join(history_snippets[-3:])}" if history_snippets else ""

    # ============================================================
    # LLM CALL (With memory context)
    # ============================================================

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

    system_prompt = (
        f"You are Zenark. User said: '{{text}}'.{history_context}\n\n"
        f"Dataset Context:\n{dataset_context}\n\n"
        "Output Guide:\n"
        "• 2–3 sentences.\n"
        "• Reinforce positive direction.\n"
        "• Reference the user's current message and recent history.\n"
        "• End with exactly one playful question.\n"
        "• Under 100 words.\n"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt.format(text=text)),
        HumanMessage(content=text)
    ])

    out = response.content
    return out if isinstance(out, str) else str(out)

@tool
async def negative_conversation_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle negative emotions with concise, AI-generated empathy."""

    # --- Dataset scoring ---
    def score(item: dict) -> int:
        if not isinstance(item, dict):
            return 0
        q = set(re.findall(r"\w+", text.lower()))
        dumped = json.dumps(item).lower()
        iwords = set(re.findall(r"\w+", dumped))
        base = len(q & iwords)
        bonus = 10 if item.get("category") in NEG_CATEGORIES else 0
        return base + bonus

    best = max(NEG_DATA, key=score, default=None)

    if best and isinstance(best, dict):
        dataset_context = f"Category: {best.get('category', 'general')}"
    else:
        dataset_context = "General negative emotion—focus on validation."

    # Inject context if available
    history_context = f"\nRecent history: {'; '.join(history_snippets[-3:])}" if history_snippets else ""

    # ============================================================
    # LLM CALL (With memory context)
    # ============================================================

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    system_prompt = (
        f"You are Zenark. User said: '{{text}}'.{history_context}\n\n"
        f"Dataset Context:\n{dataset_context}\n\n"
        "Output Rules:\n"
        "• 2–3 sentences.\n"
        "• Validate without softening or motivational tone.\n"
        "• Reflect specifics from the user's message and recent history.\n"
        "• End with a single gentle and non-repetitive question.\n"
        "• Stay under 100 words.\n"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt.format(text=text)),
        HumanMessage(content=text)
    ])

    out = response.content
    return out if isinstance(out, str) else str(out)

@tool
async def marks_tool(text: str, student_id: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Fetch marks and generate a memory-aware response."""

    # Inject context if available
    history_context = f"\nRecent history: {'; '.join(history_snippets[-2:])}" if history_snippets else ""

    # ============================================================
    # FETCH MARKS FROM MONGO
    # ============================================================
    try:
        actual_id = ObjectId(student_id)

        if marks_col is None:
            raise RuntimeError("MongoDB marks_col not initialized")

        doc = await marks_col.find_one({"_id": actual_id})

    except Exception:
        doc = None


    # ============================================================
    # NO MARKS → PURE EMOTIONAL SUPPORT
    # ============================================================
    if not doc:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = (
            f"You are Zenark.{history_context}\n"
            "User said: '{{text}}'.\n"
            "Respond in 2–3 sentences.\n"
            "Validate stress. No advice. End with one open question.\n"
        )
        r = await llm.ainvoke([
            SystemMessage(content=prompt.format(text=text)),
            HumanMessage(content=text)
        ])
        out = r.content
        return out if isinstance(out, str) else str(out)

    # ============================================================
    # PROCESS MARKS
    # ============================================================
    subjects = {
        'Physics': ['Physics on 1st PU Syllabus', 'Physics On 2nd PU Syllabus'],
        'Chemistry': ['Chemistry on 1st PU Syllabus', 'Chemistry On 2nd PU Syllabus'],
        'Botany': ['Botany on 1st PU Syllabus', 'Botany On 2nd PU Syllabus'],
        'Zoology': ['Zoology on 1st PU Syllabus', 'Zoology On 2nd PU Syllabus']
    }

    totals = {}
    for subj, fields in subjects.items():
        totals[subj] = sum(extract_int(doc.get(f, 0)) for f in fields)

    existing = {k: v for k, v in totals.items() if v > 0}

    if not existing:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = (
            f"You are Zenark.{history_context}\n"
            "User said: '{{text}}'.\n"
            "Validate stress. No advice. One open question.\n"
        )
        r = await llm.ainvoke([
            SystemMessage(content=prompt.format(text=text)),
            HumanMessage(content=text)
        ])
        out = r.content
        return out if isinstance(out, str) else str(out)

    # ============================================================
    # BUILD MARKS SUMMARY
    # ============================================================
    summary = []
    for subj, score in existing.items():
        q = " (strong!)" if score > 150 else " (weak zone)" if score < 100 else ""
        summary.append(f"{subj}: {score}{q}")

    marks_str = " | ".join(summary)
    avg = sum(existing.values()) / len(existing)
    name = doc.get("Unnamed: 2", "").strip()
    name_ref = f", {name}, " if name else " "

    # ============================================================
    # FINAL LLM RESPONSE
    # ============================================================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = (
        f"You are Zenark.{history_context}\n"
        "User said: '{{text}}'.\n\n"
        f"Their marks{name_ref}are: {marks_str} (avg: {avg:.1f}).\n\n"
        "Respond in 2–4 sentences.\n"
        "– Validate the emotional weight.\n"
        "– Mention the marks without judgment.\n"
        "– Ask exactly one gentle question.\n"
        "– No advice.\n"
    )

    r = await llm.ainvoke([
        SystemMessage(content=prompt.format(text=text)),
        HumanMessage(content=f"Marks context: {marks_str}")
    ])

    out = r.content
    return out if isinstance(out, str) else str(out)

@tool
async def exam_tips_tool(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """category: exam_tips"""  # NEW: Memori recalls past study struggles

    # Inject context if available
    history_context = f"\nRecent history (e.g., previous marks discussion): {'; '.join(history_snippets[-2:])}" if history_snippets else ""

    t = text.lower()

    # detect exam
    exam = next((key for key,_ in EXAM_TIPS if key.lower() in t), None)

    # exam-specific prompt
    if exam:
        tips = [tip for k, tip in EXAM_TIPS if k == exam]
        selected_tip = tips[0] if tips else "Maintain a consistent revision cycle."

        prompt = f"""
You are an educational and performance-oriented assistant.{history_context}
Avoid emotional wording. Avoid metaphors. Avoid soft therapy tone.

The student is preparing for: {exam}
The core recommended technique is: {selected_tip}

Write a short response that:
- states the strategy directly
- describes how to apply it
- highlights one failure pattern to avoid
- asks one clear diagnostic question about their preparation

Do not add decoration, emojis, motivation, or filler.
"""

        response = await llm_exam.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content.strip()

    # general study strategies prompt
    top_methods = ", ".join(STUDY_TECHNIQUES[:4])

    prompt = f"""
You are an exam-performance assistant.{history_context}
Avoid emotional framing and supportive tones.

The student asked for general study help.
Your output must:
- list the four scientific techniques: {top_methods}
- describe when each should be applied
- specify the typical mistake students make with these methods
- end with one direct question assessing the student's current study structure

Do not add decoration, emojis, or conversational softening.
"""

    response = await llm_exam.ainvoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip()

@tool
async def llm_generate(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Generic empathetic fallback with full MemoryManager context."""

    # Inject context if available
    history_context = f"\nRecent history: {'; '.join(history_snippets[-3:])}" if history_snippets else ""

    # ============================================================
    # LLM RESPONSE (With memory context)
    # ============================================================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    system_prompt = (
        f"You are Zenark.{history_context}\n"
        "Respond empathetically.\n\n"
        "Rules:\n"
        "• Keep it to 2–3 sentences.\n"
        "• Validate the emotion without therapy tone.\n"
        "• End with exactly one gentle exploratory question.\n"
        "• Avoid advice.\n"
        "• Avoid clinical vocabulary.\n"
    )

    r = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ])

    out = r.content
    return out if isinstance(out, str) else str(out)

@tool
async def multilingual_handler(text: str, session_id: str = "", history_snippets: List[str] = []) -> str:
    """Handle non-English Indian language inputs with cultural sensitivity."""
    detected_lang = MultilingualDetector.detect_language(text)
    
    if detected_lang:
        logging.info(f"🌐 Multilingual: Detected {detected_lang} language")
        response = await MultilingualDetector.get_multilingual_response(text, detected_lang)
        return response
    
    # Fallback to generic LLM handler if no Indian language detected
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    history_context = f"\nRecent history: {'; '.join(history_snippets[-3:])}" if history_snippets else ""
    
    system_prompt = (
        f"You are Zenark.{history_context}\n"
        "Respond empathetically.\n\n"
        "Rules:\n"
        "• Keep it to 2–3 sentences.\n"
        "• Validate the emotion without therapy tone.\n"
        "• End with exactly one gentle exploratory question.\n"
        "• Avoid advice.\n"
        "• Avoid clinical vocabulary.\n"
    )
    
    r = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ])
    
    out = r.content
    return out if isinstance(out, str) else str(out)

# ===================================================
# MULTILINGUAL SUPPORT & INTENT CLASSIFICATION
# ===================================================

class MultilingualDetector:
    """Detect Indian languages (Unicode + Romanized) and provide culturally appropriate responses"""
    
    # Indian language patterns (Unicode ranges and common words)
    INDIAN_LANGUAGE_PATTERNS = {
        'hindi': {
            'unicode_range': r'[\u0900-\u097F]',
            'common_words': ['हैं', 'है', 'मैं', 'तुम', 'क्या', 'नहीं', 'हाँ', 'कैसे'],
            'romanized_words': ['main', 'mein', 'tum', 'kya', 'nahi', 'nahin', 'haan', 'hai', 'kaise', 'kyun', 'aap', 'hum', 'mujhe', 'tumhe', 'kuch', 'sab'],
            'romanized_phrases': ['kya hal hai', 'kaise ho', 'theek hai', 'accha hai', 'namaste', 'shukriya', 'dhanyavaad'],
            'greeting': 'नमस्ते! मैं आपकी बात समझ सकता हूं। 🙏',
        },
        'tamil': {
            'unicode_range': r'[\u0B80-\u0BFF]',
            'common_words': ['நான்', 'நீ', 'என்ன', 'இல்லை', 'ஆம்'],
            'romanized_words': ['naan', 'nee', 'enna', 'illai', 'aam', 'epdi', 'yenna', 'nalla', 'vanakkam', 'nandri', 'ungal'],
            'romanized_phrases': ['epdi irukinga', 'nalla irukken', 'enna achu'],
            'greeting': 'வணக்கம்! நான் உங்களுடன் பேசலாம். 🙏',
        },
        'telugu': {
            'unicode_range': r'[\u0C00-\u0C7F]',
            'common_words': ['నేను', 'నువ్వు', 'ఏమి', 'కాదు', 'అవును'],
            'romanized_words': ['nenu', 'nuvvu', 'emi', 'kaadu', 'avunu', 'ela', 'manchidi', 'namaskaram', 'dhanyavadalu', 'meeku'],
            'romanized_phrases': ['ela unnaru', 'bagunnanu', 'emi jaruguthundi'],
            'greeting': 'నమస్కారం! నేను మీతో మాట్లాడగలను. 🙏',
        },
        'bengali': {
            'unicode_range': r'[\u0980-\u09FF]',
            'common_words': ['আমি', 'তুমি', 'কি', 'না', 'হ্যাঁ'],
            'romanized_words': ['ami', 'tumi', 'ki', 'na', 'haan', 'kemon', 'bhalo', 'namaste', 'dhonnobad', 'apni'],
            'romanized_phrases': ['kemon acho', 'bhalo achi', 'ki hoyeche'],
            'greeting': 'নমস্কার! আমি আপনার সাথে কথা বলতে পারি। 🙏',
        },
        'gujarati': {
            'unicode_range': r'[\u0A80-\u0AFF]',
            'common_words': ['હું', 'તું', 'શું', 'નથી', 'હા'],
            'romanized_words': ['hun', 'tu', 'shu', 'nathi', 'haa', 'kem', 'saras', 'namaste', 'aabhar', 'tamne'],
            'romanized_phrases': ['kem cho', 'majama chu', 'shu thayu'],
            'greeting': 'નમસ્તે! હું તમારી સાથે વાત કરી શકું છું. 🙏',
        },
        'kannada': {
            'unicode_range': r'[\u0C80-\u0CFF]',
            'common_words': ['ನಾನು', 'ನೀನು', 'ಏನು', 'ಇಲ್ಲ', 'ಹೌದು'],
            'romanized_words': ['naanu', 'neenu', 'enu', 'illa', 'haudu', 'hege', 'chennagi', 'namaskara', 'dhanyavada', 'nimma'],
            'romanized_phrases': ['hege iddeera', 'chennagi iddini', 'enu aayitu'],
            'greeting': 'ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮೊಂದಿಗೆ ಮಾತನಾಡಬಹುದು. 🙏',
        },
        'malayalam': {
            'unicode_range': r'[\u0D00-\u0D7F]',
            'common_words': ['ഞാൻ', 'നീ', 'എന്ത്', 'ഇല്ല', 'ഉണ്ട്'],
            'romanized_words': ['njan', 'njaan', 'nee', 'enthu', 'illa', 'undu', 'engane', 'nalla', 'namaskaram', 'nanni', 'ningal'],
            'romanized_phrases': ['engane undu', 'nalla undu', 'enthu patti'],
            'greeting': 'നമസ്കാരം! എനിക്ക് നിങ്ങളോട് സംസാരിക്കാം.',
        },
        'marathi': {
            'unicode_range': r'[\u0900-\u097F]',  # Shares Devanagari with Hindi
            'common_words': ['मी', 'तू', 'काय', 'नाही', 'होय'],
            'romanized_words': ['mi', 'mee', 'tu', 'kay', 'nahi', 'hoy', 'kasa', 'bara', 'namaskar', 'dhanyavad', 'tumhi'],
            'romanized_phrases': ['kasa aahes', 'bara aahe', 'kay zala'],
            'greeting': 'नमस्कार! मी तुमच्याशी बोलू शकतो. ',
        },
        'punjabi': {
            'unicode_range': r'[\u0A00-\u0A7F]',
            'common_words': ['ਮੈਂ', 'ਤੂੰ', 'ਕੀ', 'ਨਹੀਂ', 'ਹਾਂ'],
            'romanized_words': ['main', 'tu', 'ki', 'nahi', 'haan', 'kiddan', 'vadiya', 'sat sri akal', 'shukriya', 'tuhanu'],
            'romanized_phrases': ['kiddan aa', 'theek aa', 'ki hoya'],
            'greeting': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੇ ਨਾਲ ਗੱਲ ਕਰ ਸਕਦਾ ਹਾਂ। 🙏',
        }
    }
    
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """Detect if text contains Indian language (Unicode or Romanized)"""
        text_lower = text.lower()
        
        # Track language match scores
        language_scores: Dict[str, int] = {}
        
        for lang, config in MultilingualDetector.INDIAN_LANGUAGE_PATTERNS.items():
            score = 0
            
            # Check Unicode range (highest priority)
            if re.search(config['unicode_range'], text):
                return lang  # Immediate return for Unicode match
            
            # Check common Unicode words
            for word in config['common_words']:
                if word in text:
                    return lang  # Immediate return for Unicode word match
            
            # Check romanized words
            romanized_words = config.get('romanized_words', [])
            for word in romanized_words:
                # Use word boundaries to avoid partial matches (e.g., 'main' in 'remain')
                if re.search(rf'\b{re.escape(word)}\b', text_lower):
                    score += 2  # Higher weight for romanized word match
            
            # Check romanized phrases
            romanized_phrases = config.get('romanized_phrases', [])
            for phrase in romanized_phrases:
                if phrase in text_lower:
                    score += 5  # Highest weight for phrase match
            
            if score > 0:
                language_scores[lang] = score
        
        # Return language with highest score (minimum threshold: 2)
        if language_scores:
            best_lang = max(language_scores.items(), key=lambda x: x[1])[0]
            if language_scores[best_lang] >= 2:
                logging.info(f"🌐 Romanized language detected: {best_lang} (score: {language_scores[best_lang]})")
                return best_lang
        
        return None
    
    @staticmethod
    async def get_multilingual_response(text: str, detected_lang: str) -> str:
        """Generate response acknowledging language and offering English support"""
        lang_config = MultilingualDetector.INDIAN_LANGUAGE_PATTERNS.get(detected_lang, {})
        greeting = lang_config.get('greeting', 'Hello! 🙏')
        
        # Empathetic response acknowledging language preference
        response = (
            f"{greeting}\n\n"
            f"I can see you're writing in {detected_lang.capitalize()}. That's wonderful! 💙\n\n"
            f"Right now, I can understand and respond best in English, but I want you to know that "
            f"your feelings and thoughts are just as valid in any language. "
            f"If you're comfortable, please share what's on your mind in English, and I'll be here to listen and support you.\n\n"
            f"Is there something specific you'd like to talk about today?"
        )
        return response

class IntentClassifier:
    """Pre-process user input with intent classification before routing"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()
    
    @staticmethod
    def match_intent(user_text: str) -> Optional[Dict[str, Any]]:
        """Match user input against predefined intent patterns"""
        normalized_text = IntentClassifier.normalize_text(user_text)
        
        # Priority 1: Check empathic intents (mental health focused)
        for intent_obj in INTENT_DATA_EMPATHIC:
            patterns = intent_obj.get('patterns', [])
            for pattern in patterns:
                if IntentClassifier.normalize_text(pattern) == normalized_text:
                    return {
                        'dataset': 'empathic',
                        'tag': intent_obj.get('tag', ''),
                        'responses': intent_obj.get('responses', []),
                        'match_type': 'exact'
                    }
        
        # Priority 2: Check general intents
        for intent_obj in INTENT_DATA_GENERAL:
            patterns = intent_obj.get('text', [])
            for pattern in patterns:
                if IntentClassifier.normalize_text(pattern) == normalized_text:
                    return {
                        'dataset': 'general',
                        'intent': intent_obj.get('intent', ''),
                        'responses': intent_obj.get('responses', []),
                        'match_type': 'exact'
                    }
        
        # Priority 3: Partial matching for common greetings/thanks
        common_intents = {
            'greeting': ['hi', 'hello', 'hey', 'hola', 'namaste', 'vanakkam'],
            'thanks': ['thanks', 'thank you', 'thx', 'appreciate'],
            'goodbye': ['bye', 'goodbye', 'see you', 'later']
        }
        
        for intent_tag, keywords in common_intents.items():
            if any(keyword in normalized_text for keyword in keywords):
                # Find matching intent from empathic dataset
                for intent_obj in INTENT_DATA_EMPATHIC:
                    if intent_obj.get('tag') == intent_tag:
                        return {
                            'dataset': 'empathic',
                            'tag': intent_tag,
                            'responses': intent_obj.get('responses', []),
                            'match_type': 'partial'
                        }
        
        return None
    
    @staticmethod
    def get_intent_response(intent_match: Dict[str, Any]) -> str:
        """Get response for matched intent"""
        import random
        responses = intent_match.get('responses', [])
        if not responses:
            return ""
        
        # Return a random response from the matched intent
        response = random.choice(responses)
        
        # Add context about Zenark for empathic responses
        if intent_match.get('dataset') == 'empathic':
            response = response.replace('Zenark platform', 'our conversation here')
        
        return response

# ===================================================
# ROUTER CONFIGURATION (WITH MULTILINGUAL & INTENT PRE-PROCESSING)
# ===================================================

# Tool registry
TOOLS = [
    crisis_handler, substance_handler, moral_risk_handler,
    end_chat_handler, positive_conversation_handler,
    negative_conversation_handler, marks_tool, exam_tips_tool,
    multilingual_handler, llm_generate
]
TOOL_MAP = {t.name: t for t in TOOLS}

# NEW: Router LLM with Memori (injects session history for better routing)
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# FIXED: Bind tools after fixing ObjectId type issue
router_llm_with_tools = router_llm.bind_tools(TOOLS)

ROUTER_SYSTEM_PROMPT = """You are a strict routing system. Your ONLY job is to select ONE tool.

**SAFETY FIRST (Priority 1):**
- "kill myself", "end my life", "hurt myself", "suicide" → crisis_handler
- "weed", "smoking", "drugs", "alcohol", "vape", "puff" → substance_handler
- "kill someone", "hurt them", "illegal", "beat him" → moral_risk_handler
- "bye", "exit", "goodbye", "see you" → end_chat_handler

**ACADEMIC QUERIES (Priority 2):**
- "marks", "score", "percentage", "grades", "results", "physics", "chemistry", "math", "biology", "botany", "zoology", "my marks", "performance", "low marks" → marks_tool
- "JEE", "NEET", "IAS", "exam tips", "study method", "revision", "mock test", "prepare for exam" → exam_tips_tool

**EMOTIONAL STATE (Priority 3):**
- Emotion: positive → positive_conversation_handler
- Emotion: negative → negative_conversation_handler

**FALLBACK:**
- Everything else → llm_generate

**ENFORCEMENT:**
If you fail to output a tool call, you must respond exactly: "ERROR: Return a tool call only."

REMEMBER: Mental health is our focus. Even academic queries deserve empathetic handling."""

# ============================================
# ROUTER MEMORY MANAGER (STM + LTM)
# ============================================

class RouterMemory:
    """Intelligent router memory with STM (session) and LTM (persistent) capabilities"""
    
    def __init__(self, session_id: str, student_id: str, router_memory_col: AsyncIOMotorCollection):
        self.session_id = session_id
        self.student_id = student_id
        self.router_memory_col = router_memory_col
        
        # Short-term memory (current session)
        self.stm_tool_sequence: List[str] = []  # Tool usage sequence in current session
        self.stm_emotion_pattern: List[str] = []  # Emotion pattern in current session
        self.stm_topic_focus: List[str] = []  # Topic focus in current session
        
        # Long-term memory (loaded from MongoDB)
        self.ltm_tool_preferences: Dict[str, int] = {}  # Tool usage frequency across all sessions
        self.ltm_dominant_emotions: List[str] = []  # Dominant emotions across sessions
        self.ltm_recurring_topics: List[str] = []  # Recurring topics across sessions
        self.ltm_conversation_count: int = 0  # Total conversations across sessions
        
        # Contextual insights
        self.last_tool: Optional[str] = None
        self.last_emotion: Optional[str] = None
        self.conversation_flow: List[Dict[str, str]] = []  # Track conversation flow
        
    async def load_ltm(self) -> None:
        """Load long-term memory from MongoDB"""
        try:
            doc = await self.router_memory_col.find_one({
                "session_id": self.session_id,
                "student_id": self.student_id
            })
            
            if doc:
                self.ltm_tool_preferences = doc.get("tool_preferences", {})
                self.ltm_dominant_emotions = doc.get("dominant_emotions", [])
                self.ltm_recurring_topics = doc.get("recurring_topics", [])
                self.ltm_conversation_count = doc.get("conversation_count", 0)
                self.last_tool = doc.get("last_tool")
                self.last_emotion = doc.get("last_emotion")
                self.conversation_flow = doc.get("conversation_flow", [])[-10:]  # Last 10 interactions
                
                logging.info(f"📚 LTM loaded for session {self.session_id}: {self.ltm_conversation_count} conversations")
        except Exception as e:
            logging.warning(f"Failed to load LTM for session {self.session_id}: {e}")
    
    async def update_memory(self, tool_used: str, emotion: str, topic: str, user_text: str) -> None:
        """Update both STM and LTM after routing decision"""
        # Update STM
        self.stm_tool_sequence.append(tool_used)
        self.stm_emotion_pattern.append(emotion)
        self.stm_topic_focus.append(topic)
        
        # Update LTM counters
        self.ltm_tool_preferences[tool_used] = self.ltm_tool_preferences.get(tool_used, 0) + 1
        self.ltm_conversation_count += 1
        
        # Update conversation flow
        self.conversation_flow.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tool": tool_used,
            "emotion": emotion,
            "topic": topic,
            "text_snippet": user_text[:50]
        })
        
        # Keep only last 10 interactions in flow
        self.conversation_flow = self.conversation_flow[-10:]
        
        # Update last states
        self.last_tool = tool_used
        self.last_emotion = emotion
        
        # Persist to MongoDB
        try:
            await self.router_memory_col.update_one(
                {"session_id": self.session_id, "student_id": self.student_id},
                {
                    "$set": {
                        "tool_preferences": self.ltm_tool_preferences,
                        "dominant_emotions": self._calculate_dominant_emotions(),
                        "recurring_topics": self._calculate_recurring_topics(),
                        "conversation_count": self.ltm_conversation_count,
                        "last_tool": self.last_tool,
                        "last_emotion": self.last_emotion,
                        "conversation_flow": self.conversation_flow,
                        "updated_at": datetime.datetime.utcnow()
                    }
                },
                upsert=True
            )
        except Exception as e:
            logging.warning(f"Failed to persist router memory: {e}")
    
    def _calculate_dominant_emotions(self) -> List[str]:
        """Calculate dominant emotions from STM pattern"""
        from collections import Counter
        if not self.stm_emotion_pattern:
            return []
        emotion_counts = Counter(self.stm_emotion_pattern)
        return [emotion for emotion, _ in emotion_counts.most_common(3)]
    
    def _calculate_recurring_topics(self) -> List[str]:
        """Calculate recurring topics from STM focus"""
        from collections import Counter
        if not self.stm_topic_focus:
            return []
        topic_counts = Counter(self.stm_topic_focus)
        return [topic for topic, _ in topic_counts.most_common(3)]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summarized context for routing decision"""
        most_used = None
        if self.ltm_tool_preferences:
            most_used = max(self.ltm_tool_preferences.items(), key=lambda x: x[1])[0]
        
        return {
            "last_tool": self.last_tool,
            "last_emotion": self.last_emotion,
            "session_tool_sequence": self.stm_tool_sequence[-3:],  # Last 3 tools in session
            "session_emotion_pattern": self.stm_emotion_pattern[-3:],  # Last 3 emotions
            "most_used_tool": most_used,
            "dominant_emotions": self.ltm_dominant_emotions,
            "conversation_count": self.ltm_conversation_count,
            "recent_flow": self.conversation_flow[-3:]  # Last 3 interactions
        }

# ============================================
# INTELLIGENT ROUTER (CONTEXT-AWARE BRAIN)
# ============================================

class Router:
    """Intelligent router with contextual memory (STM + LTM) and adaptive decision-making"""
    
    SAFETY_PATTERNS = {
        'crisis_handler': [
            r'\b(kill|end|hurt)\s+(myself|yourself|themselves|life)\b',
            r'\bsuicid(e|al|e)\b', r'\bself.?harm\b', r'\bcut(ting)?\s+myself\b'
        ],
        'substance_handler': [
            r'\b(weed|smok(e|ing)|drug(s)?|alcohol|vape|puff|cigar(ette)?)\b'
        ],
        'moral_risk_handler': [
            r'\b(kill|hurt|beat|attack)\s+(someone|him|her|them)\b',
            r'\b(illegal|steal|rob|cheat)\b'
        ],
        'end_chat_handler': [
            r'\b(bye|goodbye|exit|quit|see you|later)\b'
        ]
    }
    
    ACADEMIC_PATTERNS = {
        'marks_tool': [
            r'\b(marks|score|grades?|results?|percentage)\b',
            r'\b(physics|chemistry|math|biology|botany|zoology)\b'
        ],
        'exam_tips_tool': [
            r'\b(JEE|NEET|IAS|BITSAT|CUET|SAT|exam|study method|revision|mock test)\b'
        ]
    }
    
    @staticmethod
    def extract_topic(text: str) -> str:
        """Extract primary topic from user text"""
        text_lower = text.lower()
        
        # Academic topics
        if re.search(r'\b(marks|score|grades?|exam|study)\b', text_lower):
            return "academic"
        # Emotional topics
        elif re.search(r'\b(sad|happy|angry|fear|anxiety|depress)\b', text_lower):
            return "emotional"
        # Crisis topics
        elif re.search(r'\b(hurt|harm|suicide|kill)\b', text_lower):
            return "crisis"
        # Substance topics
        elif re.search(r'\b(smoke|drug|alcohol|weed)\b', text_lower):
            return "substance"
        else:
            return "general"
    
    @staticmethod
    async def route_with_memory(
        text: str,
        emotion: str,
        history: List[str],
        tool_history: List[str],
        router_memory: RouterMemory,
        router_llm_with_tools: Any  # LLM with bound tools
    ) -> tuple[str, str]:
        """Intelligent LLM-based routing with contextual memory awareness"""
        text_lower = text.lower()
        
        # Get memory context
        memory_context = router_memory.get_context_summary()
        last_tool = memory_context["last_tool"]
        last_emotion = memory_context["last_emotion"]
        session_tools = memory_context["session_tool_sequence"]
        
        logging.info(f"🧠 Router Memory Context: last_tool={last_tool}, emotion_pattern={memory_context['session_emotion_pattern']}, conversation_count={memory_context['conversation_count']}")
        
        # Priority 1: Safety (ALWAYS highest priority, overrides LLM)
        for tool_name, patterns in Router.SAFETY_PATTERNS.items():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                logging.info(f"🚨 Router: {tool_name} (SAFETY OVERRIDE - bypassing LLM)")
                topic = Router.extract_topic(text)
                await router_memory.update_memory(tool_name, emotion, topic, text)
                return tool_name, text
        
        # Priority 2: LLM-based intelligent tool selection with context
        try:
            # Build context-rich prompt for LLM
            recent_history = "\n".join(history[-3:]) if history else "No previous history"
            recent_tools = ", ".join(session_tools) if session_tools else "None"
            
            system_context = f"""You are Zenark's intelligent routing system. Analyze the user's message and conversation context to select the MOST APPROPRIATE tool.

**Current Context:**
- User's emotional state: {emotion}
- Last tool used: {last_tool or 'None'}
- Recent tools: {recent_tools}
- Last emotion: {last_emotion or 'None'}
- Total conversations: {memory_context['conversation_count']}

**Recent Conversation:**
{recent_history}

**Routing Guidelines:**
1. **Academic queries** (marks, exams, study tips) → Use marks_tool or exam_tips_tool
2. **Positive emotions** (joy, happiness, celebration) → Use positive_conversation_handler
3. **Negative emotions** (sadness, anxiety, stress) → Use negative_conversation_handler
4. **Substance questions** (smoking, drugs, alcohol) → Use substance_handler
5. **Moral risks** (violence, illegal intent) → Use moral_risk_handler
6. **Goodbyes** (bye, exit) → Use end_chat_handler
7. **Crisis situations** (self-harm, suicide) → Use crisis_handler (CRITICAL)
8. **General/unclear** → Use llm_generate

**Context Awareness:**
- If user discussed marks recently and now asks about study → Use exam_tips_tool
- If user shows repeated negative emotions → Prioritize negative_conversation_handler
- If user switches topics → Adapt to new context

Select ONE tool that best matches the user's current need and conversation flow."""
            
            messages = [
                SystemMessage(content=system_context),
                HumanMessage(content=f"User message: {text}")
            ]
            
            # Invoke LLM with bound tools
            response = await router_llm_with_tools.ainvoke(messages)
            
            # Extract tool calls from response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                selected_tool_call = response.tool_calls[0]
                tool_name = selected_tool_call['name']
                
                logging.info(f"🤖 Router: {tool_name} (LLM-selected based on context)")
                
                topic = Router.extract_topic(text)
                await router_memory.update_memory(tool_name, emotion, topic, text)
                return tool_name, text
            else:
                # Fallback: No tool call returned, use emotion-based routing
                logging.warning("⚠️ Router: LLM did not return tool call, using fallback")
                if emotion == "positive":
                    tool_name = "positive_conversation_handler"
                elif emotion == "negative":
                    tool_name = "negative_conversation_handler"
                else:
                    tool_name = "llm_generate"
                
                topic = Router.extract_topic(text)
                await router_memory.update_memory(tool_name, emotion, topic, text)
                return tool_name, text
                
        except Exception as e:
            logging.error(f"❌ Router: LLM routing failed: {e}")
            
            # Fallback to regex-based routing
            logging.info("🔄 Router: Falling back to regex-based routing")
            
            # Academic patterns
            for tool_name, patterns in Router.ACADEMIC_PATTERNS.items():
                if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                    logging.info(f"🔍 Router: {tool_name} (FALLBACK regex match)")
                    topic = Router.extract_topic(text)
                    await router_memory.update_memory(tool_name, emotion, topic, text)
                    return tool_name, text
            
            # Emotion-based fallback
            if emotion == "positive":
                tool_name = "positive_conversation_handler"
            elif emotion == "negative":
                tool_name = "negative_conversation_handler"
            else:
                tool_name = "llm_generate"
            
            logging.info(f"😊 Router: {tool_name} (FALLBACK emotion-based)")
            topic = Router.extract_topic(text)
            await router_memory.update_memory(tool_name, emotion, topic, text)
            return tool_name, text
    
    @staticmethod
    def route(text: str, emotion: str) -> tuple[str, str]:
        """Legacy sync route method (kept for backward compatibility) - use route_with_memory() for intelligent routing"""
        text_lower = text.lower()

        # Priority 1: Safety (regex matching) - Overrides context
        for tool_name, patterns in Router.SAFETY_PATTERNS.items():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                logging.info(f"🔍 Router: {tool_name} (safety match)")
                return tool_name, text

        # Priority 2: Academic
        for tool_name, patterns in Router.ACADEMIC_PATTERNS.items():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                logging.info(f"🔍 Router: {tool_name} (academic match)")
                return tool_name, text

        # Priority 3: Emotion
        if emotion == "positive":
            logging.info(f"🔍 Router: positive_conversation_handler")
            return "positive_conversation_handler", text
        elif emotion == "negative":
            logging.info(f"🔍 Router: negative_conversation_handler")
            return "negative_conversation_handler", text

        # Fallback
        logging.info(f"🔍 Router: llm_generate (fallback)")
        return "llm_generate", text

# ============================================
# GRAPH NODES (WITH ROUTER MEMORY INTEGRATION)
# ============================================

async def router_node(state: GraphState) -> Dict[str, Any]:
    """Route user input with intent classification pre-processing and intelligent memory-aware router"""
    text = state["user_text"]
    session_id = state["session_id"]
    student_id = state["student_id"]
    
    # Extract recent history for context
    history = state.get("history_snippets", [])
    tool_history = state.get("tool_history", [])
    
    # ============================================================
    # PRIORITY 0: MULTILINGUAL DETECTION
    # ============================================================
    detected_lang = MultilingualDetector.detect_language(text)
    if detected_lang:
        logging.info(f"🌐 Router: Multilingual detected ({detected_lang}) - routing to multilingual_handler")
        return {
            "emotion": "neutral",
            "selected_tool": "multilingual_handler",
            "tool_input": text,
            "tool_history": tool_history + ["multilingual_handler"],
            "debug_info": {
                "emotion": "neutral",
                "tool": "multilingual_handler",
                "detected_language": detected_lang,
                "previous_tool": tool_history[-1] if tool_history else None
            }
        }
    
    # ============================================================
    # PRIORITY 1: INTENT CLASSIFICATION PRE-PROCESSING
    # ============================================================
    intent_match = IntentClassifier.match_intent(text)
    if intent_match:
        intent_response = IntentClassifier.get_intent_response(intent_match)
        logging.info(f"🎯 Router: Intent match found - {intent_match.get('tag') or intent_match.get('intent')} ({intent_match.get('match_type')})")
        
        # Return intent response directly (bypass normal routing)
        return {
            "emotion": "neutral",
            "selected_tool": "intent_classifier",  # Special marker
            "tool_input": text,
            "tool_history": tool_history + ["intent_classifier"],
            "final_output": intent_response,  # Set response directly
            "debug_info": {
                "emotion": "neutral",
                "tool": "intent_classifier",
                "intent_match": intent_match,
                "previous_tool": tool_history[-1] if tool_history else None
            }
        }
    
    # ============================================================
    # PRIORITY 2: NORMAL EMOTION DETECTION & ROUTING
    # ============================================================
    emotion = emotion_detector.detect(text)
    
    # Initialize RouterMemory for this request
    if router_memory_col is None:
        raise RuntimeError("Router memory collection not initialized")
    
    router_memory = RouterMemory(session_id, student_id, router_memory_col)
    await router_memory.load_ltm()  # Load long-term memory from MongoDB
    
    # Use intelligent routing with memory
    tool_name, tool_input = await Router.route_with_memory(
        text=text,
        emotion=emotion,
        history=history,
        tool_history=tool_history,
        router_memory=router_memory,
        router_llm_with_tools=router_llm_with_tools  # Pass LLM with bound tools
    )
    
    return {
        "emotion": emotion,
        "selected_tool": tool_name,
        "tool_input": tool_input,
        "tool_history": tool_history + [tool_name],
        "debug_info": {
            "emotion": emotion,
            "tool": tool_name,
            "previous_tool": tool_history[-1] if tool_history else None,
            "memory_context": router_memory.get_context_summary()
        }
    }

async def tool_dispatch(state: GraphState) -> Dict[str, Any]:
    """Execute selected tool with proper arguments (handles intent classifier special case)"""
    tool_name = state["selected_tool"]
    text = state["tool_input"]
    session_id = state["session_id"]
    student_id = state["student_id"]
    history_snippets = state["history_snippets"]
    
    # Special case: Intent classifier already set response in router_node
    if tool_name == "intent_classifier":
        final_output = state.get("final_output", "")
        logging.info(f"✅ Intent Response: {final_output[:100]}...")
        return {"final_output": final_output}
    
    tool_func = TOOL_MAP.get(tool_name, llm_generate)
    logging.info(f"⚙️ Executing: {tool_name}")
    
    # Build kwargs based on tool signature - all get session_id and history_snippets
    kwargs: Dict[str, Any] = {
        "text": text,
        "session_id": session_id,
        "history_snippets": history_snippets
    }
    if tool_name == "marks_tool":
        kwargs["student_id"] = student_id  # Already str
    
    result = await tool_func.ainvoke(kwargs)
    return {"final_output": result}

# ===================================================
# LANGGRAPH CONSTRUCTION
# ===================================================

def build_graph() -> Any:
    """Build and compile the LangGraph (called once at startup)."""
    
    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("tool_executor", tool_dispatch)
    graph.set_entry_point("router")
    graph.add_edge("router", "tool_executor")
    graph.add_edge("tool_executor", END)

    # -------------------------------------------------------
    # Use in-memory saver for checkpoints (stateless per request; scalable with horizontal scaling)
    # For distributed persistence, consider langgraph-checkpoint-mongo or Redis
    # -------------------------------------------------------
    checkpointer = MemorySaver()

    # -------------------------------------------------------
    # Compile graph
    # -------------------------------------------------------
    return graph.compile(checkpointer=checkpointer)

# ===================================================
# GENERATE RESPONSE FUNCTION (Async, Reusable) - CACHED FOR TOKEN SAVINGS
# ===================================================

def make_cache_key(*args, **kwargs) -> str:
    """
    Custom key builder for caching: Uses session_id, student_id, and hashed user_text.
    Handles both positional and keyword args to avoid multiple value errors.
    """
    import hashlib
    # Extract user_text (first arg or from kwargs)
    user_text = args[0] if len(args) > 0 and isinstance(args[0], str) else kwargs.get('user_text', '')
    # Ensure it's a string
    if not isinstance(user_text, str):
        user_text = str(user_text) if user_text is not None else ''
    # Extract session_id
    session_id = args[1] if len(args) > 1 and isinstance(args[1], str) else kwargs.get('session_id', 'anon')
    if not isinstance(session_id, str):
        session_id = str(session_id) if session_id is not None else 'anon'
    # Extract student_id
    student_id = args[2] if len(args) > 2 else kwargs.get('student_id', 'anon')
    if not isinstance(student_id, str):
        student_id = str(student_id) if student_id is not None else 'anon'
    # Ignore name
    return f"zenark_resp:{session_id}:{student_id}:{hashlib.md5(user_text.encode('utf-8')).hexdigest()}"

# The @cached and generate_response remain the same
@cached(ttl=600, cache=Cache.MEMORY, key_builder=make_cache_key)
async def generate_response(user_text: str, session_id: str) -> str:
    """
    Generate AI response using LangGraph pipeline.
    Cached by session_id + student_id + hashed user_text (TTL: 10 min) to save tokens on repeats.
    Scalable: Async, shared compiled_graph, per-request MemorySaver.
    """
    global compiled_graph
    if compiled_graph is None:
        raise RuntimeError("Graph not compiled. Call init_app first.")

    # ---------------------------------------------
    # INIT MONGODB-BASED MEMORY (Per request)
    # ---------------------------------------------
    mongo_memory = AsyncMongoChatMemory(session_id, cast(AsyncIOMotorCollection, chats_col))

    # Wait for load to complete (non-blocking, but ensure ready)
    await mongo_memory._load_existing()

    # ---------------------------------------------
    # BUILD SHORT HISTORY SNIPPETS FROM MONGODB HISTORY
    # ---------------------------------------------
    history = mongo_memory.get_history()
    history_snippets = build_history_snippets(history, limit=6)

    # ---------------------------------------------
    # GRAPH INPUT
    # ---------------------------------------------
    state_input = {
        "user_text": user_text,
        "emotion": "",
        "selected_tool": "",
        "tool_input": "",
        "final_output": "",
        "debug_info": {},
        "tool_history": mongo_memory.get_tool_history(),
        "session_id": session_id,
        "history_snippets": history_snippets
    }

    config = {"configurable": {"thread_id": session_id}}

    # ---------------------------------------------
    # RUN GRAPH (Async, concurrent-safe)
    # ---------------------------------------------
    result = await compiled_graph.ainvoke(state_input, config)

    selected_tool = result.get("selected_tool", "")
    output = result.get("final_output", "")

    # Save tool usage to memory
    await mongo_memory.append_tool(selected_tool)

    return output


async def save_conversation(
    conversation: list,
    user_name: Optional[str],
    session_id: str = "default",
    id: Optional[str] = None,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save the complete conversation record to MongoDB asynchronously.
    
    This function is the PERSISTENCE layer for all user conversations. It:
    1. Takes all conversation turns (user messages + AI responses)
    2. Analyzes emotions (optional, currently commented out)
    3. Inserts complete record into MongoDB via Motor
    4. Creates backup JSON file for disaster recovery
    5. Returns sanitized record (ObjectId → str) for JSON response
    
    MOTOR ASYNC DESIGN:
    - Uses Motor's async insert_one() to avoid blocking
    - Non-blocking I/O: Multiple conversations can be saved concurrently
    - Database connection pooling handled by Motor automatically
    
    DUAL PERSISTENCE STRATEGY:
    1. PRIMARY: MongoDB (fast, queryable, indexed for reports)
    2. BACKUP: Local JSON files (disaster recovery, auditing)
    - If MongoDB fails: File still persists for manual recovery
    - If both fail: Function logs exception but doesn't crash app
    
    INTEGRATION WITH STREAMLIT:
    - Called from zenark_streamlit_ui.py's save_conversation() wrapper
    - Wrapper uses asyncio.run_coroutine_threadsafe() for non-blocking save
    - Streamlit doesn't wait for database; save happens in background
    
    Args:
        conversation: List of dict representing conversation turns
                     Format: [{"user": "...", "ai": "..."}, ...]
        user_name: User's display name (stored as "name" field in DB)
        session_id: Unique conversation session identifier
        id: Optional MongoDB ObjectId (string) for linking to user profile
        token: Optional auth token (stored for audit trail)
    
    Returns:
        Dict[str, Any] with:
        - "name": str - User name
        - "conversation": list - Full conversation history
        - "timestamp": str - ISO format timestamp (converted from datetime)
        - "_id": str - MongoDB document ID (ObjectId as string)
        - "userId": str or None - Linked user ID
        
    Raises:
        No exceptions; all errors logged and handled gracefully.
    """
    logger.info(f"💾 Saving conversation for user={user_name} | session={session_id}")
    
    # EMOTION ANALYSIS: Optional enrichment of conversation (currently disabled)
    # Could re-enable to add emotion scores to each user message
    # Disabled because: (1) adds latency, (2) not needed for current UI, (3) available in reports
    analyzed_conversation = []
    for turn in conversation:
        # TODO: Emotion classification can be re-enabled here if needed
        # if "user" in turn:
        #     text = turn["user"]
        #     try:
        #         scores = emotion_classifier(text)  # HF pipeline call
        #         if isinstance(scores, list) and isinstance(scores[0], list):
        #             scores = scores[0]
        #         emotion_dict = {item.get("label", "unknown"): round(item.get("score", 0.0), 4) for item in scores if isinstance(item, dict)}
        #         turn["emotion_scores"] = emotion_dict
        #     except Exception as e:
        #         logger.warning(f"Emotion classification failed: {e}")
        #         turn["emotion_scores"] = {"error": str(e)}
        analyzed_conversation.append(turn)

    # RECORD CONSTRUCTION: Build document for MongoDB insertion
    user_id = ObjectId(id) if id else None  # Convert string ID to MongoDB ObjectId
    record = {
        "name": user_name or "Unknown",  # User name (indexed for queries)
        "conversation": analyzed_conversation,  # Full conversation array
        "timestamp": datetime.datetime.now(),  # UTC timestamp
        "userId": user_id,  # Link to user profile (if available)
        "token": token  # Auth token (optional, for audit)
    }

    # MOTOR ASYNC INSERT: Non-blocking insertion into MongoDB
    # Motor handles connection pooling; multiple concurrent inserts work seamlessly
    try:
        if chats_col is None:
            raise ValueError("❌ MongoDB collection (chats_col) not initialized. Call init_db() first.")
        result = await chats_col.insert_one(record)  # AWAIT: Non-blocking insert
        record["_id"] = result.inserted_id  # Capture MongoDB-generated ID
        logger.info(f"✅ Conversation inserted to MongoDB with _id={result.inserted_id}")
    except Exception as e:
        logger.exception(f"❌ Motor insert failed: {e}")
        record["_id"] = None  # Fallback: still persist file (see below)

    # BACKUP FILE PERSISTENCE: Disaster recovery strategy
    # If MongoDB is down/unreachable: conversation still saved locally
    # Files can be manually imported/recovered later
    folder = "chat_sessions"
    os.makedirs(folder, exist_ok=True)  # Create folder if needed
    path = os.path.join(
        folder,
        f"{user_name or 'unknown'}_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )  # Filename includes name + timestamp for easy identification
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            # safe_json: Custom serializer that handles datetime, ObjectId, etc.
            json.dump(record, f, indent=2, ensure_ascii=False, default=safe_json)
        logger.info(f"✅ Conversation backed up to {path}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to write backup file: {e}")  # Non-fatal

    return cast(Dict[str, Any], convert_objectid(record))  # JSON-serializable dict



# ===================================================
# FASTAPI APP
# ===================================================

app = FastAPI(title="Zenark Mental Health Bot API", version="1.0.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    token: Optional[str] = None

class SaveRequest(BaseModel):
    conversation: List[Dict]
    name: Optional[str] = None
    session_id: str = "default"
    token: Optional[str] = None

class ReportRequest(BaseModel):
    name: Optional[str] = None
    token: str
    session_id: str = "default"


# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def decode_jwt(token):
    """Decode JWT token to extract payload."""
    header, payload, signature = token.split('.')
    padded_payload = payload + '=' * (-len(payload) % 4)
    decoded_bytes = base64.urlsafe_b64decode(padded_payload)
    payload_data = json.loads(decoded_bytes)
    return payload_data

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Chat endpoint: Handles user messages asynchronously.
    Scalable for 1M+ users: Async I/O, connection pooling, horizontal scaling ready.
    """
    try:
        session_id = chat_request.session_id  # Required, no default
        user_text = chat_request.message
        token = chat_request.token  # From body

        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        if not token:
            raise HTTPException(status_code=400, detail="Token is required from frontend.")

        # NEW: Decode JWT to extract student_id
        payload = decode_jwt(token)
        student_id = payload.get('sub')  # Assuming 'sub' claim holds student_id
        if not student_id:
            raise HTTPException(status_code=401, detail="Token missing 'sub' (student_id) claim.")

        # ---------------------------------------------
        # INIT MONGODB-BASED MEMORY (Per request)
        # ---------------------------------------------
        mongo_memory = AsyncMongoChatMemory(session_id, cast(AsyncIOMotorCollection, chats_col))

        # ---------------------------------------------
        # SAVE USER MESSAGE TO MONGODB
        # ---------------------------------------------
        await mongo_memory.append_user(user_text)

        # ---------------------------------------------
        # Generate the AI response using your existing pipeline
        # ---------------------------------------------
        ai_response = await generate_response(
            user_text=user_text,
            session_id=session_id,
            student_id=student_id  # NEW: Pass extracted student_id
        )

        # ---------------------------------------------
        # SAVE AI RESPONSE TO MONGODB
        # ---------------------------------------------
        await mongo_memory.append_ai(ai_response)

        return JSONResponse(content={"response": ai_response, "session_id": session_id})

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}

@app.get("/router-memory/{session_id}/{student_id}")
async def get_router_memory(session_id: str, student_id: str):
    """Get router memory context for a user session (for debugging/insights)"""
    try:
        if router_memory_col is None:
            raise HTTPException(status_code=500, detail="Router memory not initialized")
        
        router_memory = RouterMemory(session_id, student_id, router_memory_col)
        await router_memory.load_ltm()
        
        context = router_memory.get_context_summary()
        
        return JSONResponse(content={
            "session_id": session_id,
            "student_id": student_id,
            "memory_context": context,
            "stm": {
                "tool_sequence": router_memory.stm_tool_sequence,
                "emotion_pattern": router_memory.stm_emotion_pattern,
                "topic_focus": router_memory.stm_topic_focus
            },
            "ltm": {
                "tool_preferences": router_memory.ltm_tool_preferences,
                "dominant_emotions": router_memory.ltm_dominant_emotions,
                "recurring_topics": router_memory.ltm_recurring_topics,
                "conversation_count": router_memory.ltm_conversation_count
            }
        })
    except Exception as e:
        logging.error(f"Error retrieving router memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_chat")
async def save_chat_endpoint(req: SaveRequest):
    """Save conversation endpoint."""
    print(req)
    jwt_token = req.token
    payload = decode_jwt(jwt_token)
    user_id = payload.get("id")
    record = await save_conversation(req.conversation, req.name, req.session_id, user_id, jwt_token)
    return JSONResponse(content=jsonable_encoder(record))

# ===================================================
# NEW ENDPOINT: SCORE CONVERSATION (Global Distress Score 1-10)
# ===================================================


@app.post("/score_conversation")
async def score_conversation(request: Request):
    """
    Analyze an entire conversation history and return a Global Distress Score (1–10)
    following the action_scoring_guidelines.
    """
    data = await request.json()
    session_id = data.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    # Ensure MongoDB is initialized
    if chats_col is None:
        raise HTTPException(status_code=500, detail="MongoDB not initialized")

    # Load the full session conversation (async)
    memory = AsyncMongoChatMemory(session_id, chats_col)
    await memory._load_existing()  # Ensure history is loaded
    history = memory.get_history()

    if not history.messages:
        # NEW: Handle empty history gracefully (no distress)
        logging.info(f"📊 Score: No history for {session_id} → Default score 1 (minimal distress)")
        return JSONResponse(content={"session_id": session_id, "global_distress_score": 1})

    # Summarize the conversation into plain text
    conversation_summary = "\n".join(
        [f"{'USER' if isinstance(msg, HumanMessage) else 'ZENARK'}: {msg.content}" for msg in history.messages]
    )

    # Build the scoring prompt
    scoring_prompt = f"""{action_scoring_guidelines}

Conversation SUMMARY:
{conversation_summary[:4000]}  # Truncate for token limits (adjust as needed)

Return only a single integer (1–10) as the Global Distress Score."""

    # Low-temperature deterministic LLM call (async)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = await llm.ainvoke([HumanMessage(content=scoring_prompt)])  # Use ainvoke for async

    # Extract numeric score safely
    try:
        # Normalize model output into plain text
        content = result.content if hasattr(result, 'content') else str(result)
        if isinstance(content, list):
            # Flatten if the model returned tokens or dicts
            content = " ".join(
                [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in content]
            )
        elif not isinstance(content, str):
            content = str(content)

        score_text = content.strip()
        
        # Improved extraction: Look for digits in 1-10 range
        import re
        score_match = re.search(r'\b(\d{1,2})\b', score_text)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 10:
                logging.info(f"📊 Score for {session_id}: {score}")
                return JSONResponse(content={"session_id": session_id, "global_distress_score": score})
        
        raise ValueError("No valid score found")
    except Exception as e:
        logging.error(f"Score extraction failed: {e}, Raw: {result.content}")
        # NEW: Fallback to neutral score on extraction failure
        logging.warning(f"📊 Fallback score for {session_id}: 5 (due to processing error)")
        return JSONResponse(content={"session_id": session_id, "global_distress_score": 5, "warning": "Fallback score due to processing error"})

# Vercel entrypoint (wraps your app)
# ASGI-compliant handler (Vercel calls this)
async def handler(scope, receive, send):
    """
    Vercel entrypoint: Forwards to FastAPI as ASGI.
    Fixes the 'receive/send' error by using full ASGI spec.
    """
    await app(scope, receive, send)  # Direct ASGI call—no Request wrapper needed

# Export for Vercel (they look for 'handler')
__all__ = ["handler"]
    
# ===================================================
# RUN SERVER
# ===================================================

if __name__ == "__main__":
    # Production: Use gunicorn + uvicorn workers for concurrency
    # e.g., gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run(app, host="0.0.0.0", port=8000)