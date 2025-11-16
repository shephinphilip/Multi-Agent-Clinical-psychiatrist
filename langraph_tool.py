# ============================================
# ZENARK MENTAL HEALTH BOT - PRODUCTION FASTAPI
# SCALABLE FOR 1M+ USERS WITH ASYNC MONGODB & LANGGRAPH
# WITH IN-MEMORY CACHING LAYER TO SAVE TOKENS
# ============================================

import os
import re
import json
import asyncio
import hashlib
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
from pydantic import BaseModel
import uvicorn
from aiocache import Cache, cached

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Init in-memory cache for token savings (TTL: 10 min, scalable with horizontal if needed)
cache = Cache(Cache.MEMORY)

# ===================================================
# CONFIGURATION
# ===================================================

# Load secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# MongoDB (for student marks and chat sessions)
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "pqe_db"

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

# Global graph (compiled once at startup for scalability)
compiled_graph = None

async def init_db() -> None:
    """Initialize MongoDB collections asynchronously using Motor (called at startup)."""
    global client, chats_col, marks_col
    try:
        # Motor uses built-in connection pooling for high concurrency (1M+ users)
        client = AsyncIOMotorClient(CONFIG.mongo_uri, maxPoolSize=200, minPoolSize=10)
        db = client[DB_NAME]

        chats_col = db["chat_sessions"]
        marks_col = db["student_marks"]

        # Create indexes for fast queries (O(1) lookups for sessions/marks)
        await chats_col.create_index([("session_id", 1)], unique=True, sparse=True)
        await marks_col.create_index([("_id", 1)])

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

# ===================================================
# ROUTER CONFIGURATION (SIMPLIFIED: MongoDB handles memory injection)
# ===================================================

# Tool registry
TOOLS = [
    crisis_handler, substance_handler, moral_risk_handler,
    end_chat_handler, positive_conversation_handler,
    negative_conversation_handler, marks_tool, exam_tips_tool,
    llm_generate
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
# ROUTER
# ============================================

class Router:
    """Enhanced router with regex patterns and session context awareness"""
    
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
    def route(text: str, emotion: str, history: List[str], tool_history: List[str] = []) -> tuple[str, str]:
        """Return (tool_name, tool_input) - Context-aware with previous tool and history"""
        text_lower = text.lower()
        previous_tool = tool_history[-1] if tool_history else None

        # Priority 1: Safety (regex matching) - Overrides context
        for tool_name, patterns in Router.SAFETY_PATTERNS.items():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                logging.info(f"🔍 Router: {tool_name} (safety match)")
                return tool_name, text

        # Context-aware academic routing
        if previous_tool == "marks_tool" and any(re.search(p, text_lower, re.IGNORECASE) for p in Router.ACADEMIC_PATTERNS['exam_tips_tool']):
            logging.info(f"🔍 Router: exam_tips_tool (follow-up to marks_tool)")
            return "exam_tips_tool", text

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
# GRAPH NODES
# ============================================

async def router_node(state: GraphState) -> Dict[str, Any]:
    """Route user input to appropriate tool"""
    text = state["user_text"]
    emotion = emotion_detector.detect(text)
    
    # Extract recent history for context
    history = state.get("history_snippets", [])
    tool_history = state.get("tool_history", [])
    
    tool_name, tool_input = Router.route(text, emotion, history, tool_history)
    
    return {
        "emotion": emotion,
        "selected_tool": tool_name,
        "tool_input": tool_input,
        "tool_history": tool_history + [tool_name],
        "debug_info": {"emotion": emotion, "tool": tool_name, "previous_tool": tool_history[-1] if tool_history else None}
    }

async def tool_dispatch(state: GraphState) -> Dict[str, Any]:
    """Execute selected tool with proper arguments"""
    tool_name = state["selected_tool"]
    text = state["tool_input"]
    session_id = state["session_id"]
    student_id = state["student_id"]
    history_snippets = state["history_snippets"]
    
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

# Replace the make_cache_key function with this:

# Replace the make_cache_key function with this:

# Replace the make_cache_key function with this robust version:

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
async def generate_response(user_text: str, session_id: str, student_id: Optional[str] = None, name: Optional[str] = None) -> str:
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
        "student_id": student_id or "",
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

# ===================================================
# FASTAPI APP
# ===================================================

app = FastAPI(title="Zenark Mental Health Bot API", version="1.0.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "anonymous"
    student_id: Optional[str] = None  # From frontend
    name: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Chat endpoint: Handles user messages asynchronously.
    Scalable for 1M+ users: Async I/O, connection pooling, horizontal scaling ready.
    """
    try:
        session_id = chat_request.session_id or str(uuid.uuid4())
        user_text = chat_request.message
        student_id = chat_request.student_id  # Required from frontend, None if not provided

        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        if not student_id:
            raise HTTPException(status_code=400, detail="student_id is required from frontend.")

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
            student_id=student_id,
            name=chat_request.name
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

# ===================================================
# RUN SERVER
# ===================================================

if __name__ == "__main__":
    # Production: Use gunicorn + uvicorn workers for concurrency
    # e.g., gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run(app, host="0.0.0.0", port=8000)