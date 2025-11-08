# app.py
import os, re, json, torch, datetime, logging, numpy as np
from typing import Annotated, Optional, List, Dict, Any, cast, TypedDict
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from huggingface_hub import login
from bson import ObjectId
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,SystemMessage
from langchain_classic.chains import LLMChain
from logging.handlers import RotatingFileHandler
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.encoders import jsonable_encoder
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, END
import operator
from autogen_report import generate_autogen_report
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from prompt import SYSTEM_PROMPT, USER_PROMPT, EMOTION_TIPS,detect_end_intent,detect_moral_risk,ZEN_MODE_SUGGESTION,detect_self_harm

# app = FastAPI(title="Zenark Mental Health API", version="2.0", description="Empathetic AI counseling system with detailed logging.")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or ["http://localhost:8501"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ============================================================
#  LOGGING CONFIGURATION (Unchanged)
# ============================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "zenark_server.log")

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

logging.basicConfig(encoding='utf-8', level=logging.INFO, force=True)
logger = logging.getLogger("zenark")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================
#  ENVIRONMENT + INITIALIZATION (Unchanged)
# ============================================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger.info("Server startup initiated.")
logger.info(f"Environment loaded. HF_TOKEN={'set' if HF_TOKEN else 'missing'} | MONGO_URI={'set' if MONGO_URI else 'missing'}")

# ============================================================
#  DATABASE CONNECTION (Unchanged)
# ============================================================
try:
    client = MongoClient(MONGO_URI)
    db = client["zenark_db"]
    marks_col = db["student_marks"]
    chats_col = db["chat_sessions"]
    reports_col = db["reports"]  # Added for reports
    logger.info("MongoDB connection established.")
except Exception as e:
    logger.exception(f"MongoDB connection failed: {e}")
    raise

# ============================================================
#  HUGGINGFACE LOGIN & MODEL LOADING (Unchanged)
# ============================================================
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("Logged into Hugging Face Hub successfully.")
    except Exception as e:
        logger.warning(f"Hugging Face login failed: {e}")
else:
    logger.warning("No HUGGINGFACE_TOKEN found in environment.")

logger.info("Loading embedding models and emotion classifier...")
try:
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        token=HF_TOKEN,
        top_k=None,
    )
    sentiment_classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",  # <-- Replace here
        token=HF_TOKEN,
        device=-1  # CPU-only
    )
    logger.info("All models successfully initialized.")
except Exception as e:
    logger.exception(f"Model loading failed: {e}")
    raise

# ------------------------------------------------------------------
# 1️⃣  Metadata collector (updated to include extra fields from positive.json)
# ------------------------------------------------------------------

# --------------------------------------------------------------
#  LOAD / PRE‑PROCESS THE DATASET
# --------------------------------------------------------------
def metadata_func(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the fields we will need later and store them in the Document
    metadata.  The loader will keep the *raw record* as page_content (a JSON
    string) because we set `text_content=False`.  We only need the three
    empathy fields for the vector store, so we also copy them into metadata.
    Updated to include patient_context and psychiatrist_response for positive.json.
    """
    metadata.update({
        "id":                record.get("id"),
        "category":          record.get("category"),
        "system_prompt":     record.get("system_prompt", ""),
        "patient_context":   record.get("patient_context", ""),
        "psychiatrist_response": record.get("psychiatrist_response", ""),
        "empathic_question":record.get("empathic_question", ""),
        "empathic_response":record.get("empathic_response", ""),
        "next_question":    record.get("next_question", ""),
    })
    return metadata


def embedding_content(record: Dict[str, Any]) -> str:
    """
    Produce the short string that will be embedded.
    We join the empathy fields with a pipe separator. Include patient_context and
    psychiatrist_response if present (from positive.json) for richer embeddings.
    """
    parts = [
        record.get("patient_context", ""),
        record.get("psychiatrist_response", ""),
        record.get("empathic_question", ""),
        record.get("empathic_response", ""),
        record.get("next_question",      ""),
    ]
    return " | ".join(p for p in parts if p)   # drop empty parts


DATA_PATH = "combined_dataset.json"

try:
    # ------------------------------------------------------------------
    # 1️⃣  Load the JSON.  `text_content=False` → loader returns the whole
    #     JSON object as a string in `doc.page_content`.  All fields we need
    #     are captured in `metadata` by `metadata_func`.
    # ------------------------------------------------------------------
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema=".dataset[]",          # each element of the top‑level list
        metadata_func=metadata_func,
        text_content=False,              # keep raw dict (as JSON string) -> we ignore it
    )
    raw_docs: List[Document] = loader.load()
    logger.info(f"Loaded {len(raw_docs)} raw empathy documents.")

    # ------------------------------------------------------------------
    # 2️⃣  Build the *embedding* string from the metadata we already have.
    # ------------------------------------------------------------------
    docs: List[Document] = []
    for doc in raw_docs:
        try:
            # All the fields we need are already in the metadata dict.
            # No need to parse `doc.page_content` – we just read from `doc.metadata`.
            record = {
                "patient_context": doc.metadata.get("patient_context", ""),
                "psychiatrist_response": doc.metadata.get("psychiatrist_response", ""),
                "empathic_question": doc.metadata.get("empathic_question", ""),
                "empathic_response": doc.metadata.get("empathic_response", ""),
                "next_question":     doc.metadata.get("next_question",      ""),
            }
            # Replace the page_content with the short pipe‑separated string.
            doc.page_content = embedding_content(record)
            docs.append(doc)
        except Exception as inner_e:
            logger.warning(
                f"Failed to process doc {doc.metadata.get('id','unknown')}: {inner_e}"
            )
            continue

    # ------------------------------------------------------------------
    # 3️⃣  sanity‑check output (you can delete these prints in prod)
    # ------------------------------------------------------------------
    if docs:
        logger.info(f"Successfully processed {len(docs)} documents into string content.")
        print("\n--- Sample document after processing --------------------------------")
        print(f"page_content (first 120 chars): {docs[0].page_content[:120]}...")
        print(f"metadata (first doc): {docs[0].metadata}")
    else:
        logger.error("No valid documents processed – check the JSON structure or jq_schema.")
        raise RuntimeError("Dataset processing yielded zero documents")

    # ------------------------------------------------------------------
    # 4️⃣  Build the FAISS vector store (embeds the short strings)
    # ------------------------------------------------------------------
    vectorstore = FAISS.from_documents(docs, embedding_function)
    vectorstore.save_local("faiss_zenark_index")
    logger.info("FAISS index built and saved with metadata.")

    # ------------------------------------------------------------------
    # 5️⃣  (Optional) Build a *corpus* for cheap cosine‑similarity fallback.
    #     This is the same text that the vector store indexed, but we prepend
    #     the category and system_prompt so you can do a quick filter later.
    # ------------------------------------------------------------------
    CORPUS = [
        f"{d.metadata.get('category','general')} | "
        f"{d.metadata.get('system_prompt','')} "
        f"{d.page_content}"
        for d in docs
    ]
    # Pre‑compute the embeddings once – useful if you ever want a pure‑numpy
    # cosine‑similarity lookup (the `retrieve_context` helper does this).
    CORPUS_EMB = torch.tensor(
        np.array([embedding_function.embed_query(text) for text in CORPUS])
    )
    logger.info("Corpus embeddings prepared.")

except Exception as e:
    # ------------------------------------------------------------------
    #  🚨  Anything that blows up in the block above ends up here.
    # ------------------------------------------------------------------
    logger.exception(f"Dataset loading failed: {e}")
    docs = []
    vectorstore = None
    CORPUS = []
    CORPUS_EMB = torch.tensor([])

# Load positive dataset
POSITIVE_DATA_PATH = "positive_conversation.json"
positive_vectorstore = None
try:
    positive_loader = JSONLoader(
        file_path=POSITIVE_DATA_PATH,
        jq_schema=".dataset[]",
        metadata_func=metadata_func,
        text_content=False,
    )
    raw_positive_docs: List[Document] = positive_loader.load()
    logger.info(f"Loaded {len(raw_positive_docs)} positive empathy documents.")

    positive_docs: List[Document] = []
    for doc in raw_positive_docs:
        try:
            record = {
                "patient_context": doc.metadata.get("patient_context", ""),
                "psychiatrist_response": doc.metadata.get("psychiatrist_response", ""),
                "empathic_question": doc.metadata.get("empathic_question", ""),
                "empathic_response": doc.metadata.get("empathic_response", ""),
                "next_question":     doc.metadata.get("next_question",      ""),
            }
            doc.page_content = embedding_content(record)
            positive_docs.append(doc)
        except Exception as inner_e:
            logger.warning(
                f"Failed to process positive doc {doc.metadata.get('id','unknown')}: {inner_e}"
            )
            continue

    if positive_docs:
        logger.info(f"Successfully processed {len(positive_docs)} positive documents into string content.")
        positive_vectorstore = FAISS.from_documents(positive_docs, embedding_function)
        positive_vectorstore.save_local("faiss_positive_index")
        logger.info("Positive FAISS index built and saved.")
    else:
        logger.error("No valid positive documents processed.")
except Exception as e:
    logger.exception(f"Positive dataset loading failed: {e}")
    positive_vectorstore = None

# ----------------------------------------------------------------------
# At this point you have:
#   - `vectorstore` (FAISS) ready for retrieval,
#   - `positive_vectorstore` for positive sentiment retrieval,
#   - `CORPUS` / `CORPUS_EMB` for fallback similarity checks,
#   - each `Document` carries the full empathy template in `.metadata`.
# ----------------------------------------------------------------------


class ZenarkResponse(BaseModel):
    validation: str = Field(description="Short validation of the user's feeling, 10‑15 words.")
    reflection: str = Field(description="A concise reflective sentence that shows you understood the user.")
    question: str = Field(description="One open‑ended question, encouraging the user to elaborate or choose a new topic.")

response_parser = JsonOutputParser(pydantic_object=ZenarkResponse)



# Dynamic category extraction
def extract_categories_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("dataset", data)
    return sorted({item.get("category", "").strip() for item in records if "category" in item})

categories = extract_categories_from_json(DATA_PATH)

# Guidelines (assuming Guideliness.py exists; fallback if not)
try:
    from Guideliness import (
        action_cognitive_guidelines, action_emotional_guidelines, action_developmental_guidelines,
        action_environmental_guidelines, action_family_guidelines, action_medical_guidelines,
        action_peer_guidelines, action_school_guidelines, action_social_support_guidelines,
        action_opinon_guidelines  # Fixed spelling: opinion
    )
    GUIDELINES = {
        "family_issues": action_family_guidelines,
        "school_stress": action_school_guidelines,
        "peer_issues": action_peer_guidelines,
        "developmental_details": action_developmental_guidelines,
        "medical_complaint": action_medical_guidelines,
        "emotional_distress": action_emotional_guidelines,
        "environmental_stress": action_environmental_guidelines,
        "cognitive_concern": action_cognitive_guidelines,
        "social_support": action_social_support_guidelines,
        "opinion": action_opinon_guidelines,
    }
except ImportError:
    logger.warning("Guidelines.py not found; using empty dict.")
    GUIDELINES = {}

# Map dataset categories to guidelines (dynamic + fixed)
CATEGORY_MAP = {cat: cat.replace("_", "_") for cat in categories}  # e.g., "peer_relations" -> "peer_relations"
CATEGORY_MAP.update({
    "anxiety": "emotional_distress", "bipolar": "emotional_distress", "depression": "emotional_distress",
    "emotional_functioning": "emotional_distress", "environmental_stressors": "environmental_stress",
    "family": "family_issues", "ocd": "emotional_distress", "peer_relations": "peer_issues",
    "ptsd": "emotional_distress", "school_stress": "school_stress", "developmental": "developmental_details",
    "medical": "medical_complaint", "emotional_distress": "emotional_distress",
    "environmental": "environmental_stress", "cognitive": "cognitive_concern", "social_support": "social_support",
    "opinion": "opinion",  # Added mapping for opinion
})


CRISIS_RESPONSE = """
I'm concerned about what you've shared. For immediate help you can:

• Call the National Suicide Prevention Helpline at +919152987821 (India) or 988 (US)
• Text "HELLO" to 741741 (Crisis Text Line, US) — we will route you to an Indian service if needed
• Call 112 (or 911) or go to the nearest emergency department.

Would you like more resources for your location?
"""


# Tools (unchanged, but simplified - no full agent)
@tool(
    args_schema=None,
    description="Retrieve academic marks from MongoDB for the given student name. Returns formatted string or 'No marks found'."
)
def get_academic_marks(student_name: str) -> str:
    """Retrieve academic marks from MongoDB for the given student."""
    record = marks_col.find_one({"name": {"$regex": f"^{student_name}$", "$options": "i"}})
    if not record:
        return f"No marks found for {student_name}."
    marks_list = record.get("marks", [])
    formatted = ", ".join(f"{m['subject']}: {m['marks']}" for m in marks_list)
    return f"Academic report for {student_name}: {formatted}"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [get_academic_marks]

# LCEL Chain for Category Classification (Updated to include 'opinion')
category_prompt = PromptTemplate(
    input_variables=["history", "user_text"],
    template="""Based on this conversation history: {history}

Latest message: "{user_text}"

Classify the **overall topic** into **one** category from: {categories}, or 'opinion' if the user is seeking the AI's opinion, advice, recommendation, suggestion, or personal view (e.g., "What do you think about X?", "Your advice on Y?", "Do you recommend Z?").
Examples:
- Family/parents/rude: 'family'
- School/teacher/marks/exams: 'environmental_stressors'
- Anger/anxiety/emotions: 'emotional_functioning' or 'anxiety'
- Friends/peers: 'peer_relations'
- Obsessions/hoarding: 'ocd'
- Trauma/flashbacks: 'ptsd'
- Mood swings: 'bipolar'
- Sadness/low mood: 'depression'
- "What's your opinion on handling arguments with friends?": 'opinion'
- "Do you have any advice for studying better?": 'opinion'

If the latest message is a single word answer like 'yes' or 'no' or 'okay' , and shows no interest in continuing the topic, switch to another category based on the context.
for example, if the user says 'yes' to a question about school stress, but has also mentioned family issues before, classify as 'family' and if the user is still not interested in the 'family' topic change to another category like 'environmental_stressors' or 'emotional_functioning'.

Output **only** the category name (e.g., 'family' or 'opinion'). No explanation or JSON."""
)

# Prepare categories list including 'opinion'
all_categories = list(categories) + ["opinion"]
partial_category_prompt = category_prompt.partial(categories=", ".join(all_categories))
category_chain = partial_category_prompt | llm | StrOutputParser()

# Direct tool call for marks (no agent)
def fetch_marks_direct(student_name: str) -> str:
    """Direct marks fetch (bypass agent to avoid errors)."""
    record = marks_col.find_one({"name": {"$regex": f"^{student_name}$", "$options": "i"}})
    if not record:
        return "null"
    marks_list = record.get("marks", [])
    formatted = ", ".join(f"{m['subject']}: {m['marks']}" for m in marks_list)
    return formatted

logger.info("LCEL category chain initialized.")

def retrieve_context(query, category, top_k=3, threshold=0.25):
    if not CORPUS:
        return []
    
    # Filter corpus by category for more relevant retrieval
    if not category:
        cat_corpus = CORPUS
    else:
        cat_corpus = [c for c in CORPUS if (category or "").lower() in c.lower()]
    if not cat_corpus:
        cat_corpus = CORPUS  # Fallback to full corpus
    
    # Embed filtered corpus
    cat_emb = np.array([embedding_function.embed_query(text) for text in cat_corpus])
    q_emb = np.array(embedding_function.embed_query(query)).reshape(1, -1)
    
    sims = cosine_similarity(q_emb, cat_emb)[0]
    idxs = sims.argsort()[-top_k:][::-1]
    contexts = [cat_corpus[i] for i in idxs if sims[i] >= threshold]
    return contexts

def get_guideline_prompt(category: str) -> str:
    mapped = CATEGORY_MAP.get(category, "emotional_distress")
    guideline = GUIDELINES.get(mapped, "")
    return f"Guideline for {category}: {guideline}" if guideline else ""

# Persistent session state for category counts
session_category_counts: Dict[str, Dict[str, int]] = {}


# --------------------------------------------------------------
# 2️⃣  Moral‑risk detection (run before we ever call the LLM)
# --------------------------------------------------------------
import re

# ============================================================
#  LANGGRAPH SETUP
# ============================================================
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_text: str
    session_id: str
    name: Optional[str]
    question_index: int
    max_questions: int  # Now per-category max (default 5)
    category: str
    last_category: Optional[str]
    category_questions_count: int
    emotion_scores: Dict[str, float]
    marks: Optional[str]
    rag_context: str
    guideline: str
    sentiment: Optional[str]
    is_crisis: bool
    pending_questions: List[str]  # New: Track unanswered questions

def sentiment_node(state: GraphState) -> Dict[str, Any]:
    recent = state["messages"][-6:]
    recent_history = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" 
        for m in recent
    ])
    combined = recent_history + "\nUser: " + state["user_text"]
    try:
        if 'sentiment_classifier' in globals():
            result = sentiment_classifier(combined)
            if isinstance(result, list) and len(result) > 0:
                top = result[0]
                label = top.get('label', '').upper()
                score = top.get('score', 0.0)
                if score < 0.6:  # <-- Add threshold; adjust as needed (e.g., 0.6 for balanced sensitivity)
                    sentiment = "neutral"  # Or fallback to "negative" if you prefer caution
                else:
                    sentiment = "positive" if label == 'POSITIVE' else "negative"
            else:
                sentiment = "negative"
        else:
            sentiment = "negative"
    except Exception as e:
        logger.warning(f"Sentiment classification failed: {e}. Defaulting to negative.")
        sentiment = "negative"
    logger.info(f"Sentiment classified as: {sentiment} (label: {label if 'label' in locals() else 'N/A'}, score: {score if 'score' in locals() else 'N/A'})")  # <-- Enhanced logging
    return {"sentiment": sentiment}

def suicide_check_node(state: GraphState) -> Dict[str, Any]:
    recent = state["messages"][-6:]
    recent_history = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" 
        for m in recent
    ])
    combined = recent_history + "\nUser: " + state["user_text"]
    is_crisis = detect_self_harm(combined)
    if is_crisis:
        logger.warning("Self-harm detected in negative sentiment path – returning crisis script.")
        return {
            "messages": [AIMessage(content=CRISIS_RESPONSE)],
            "is_crisis": True
        }
    else:
        return {"is_crisis": False}

def classify_category(state: GraphState) -> Dict[str, Any]:
    global session_category_counts
    messages = state["messages"]
    recent = messages[-6:]
    recent_history = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" 
        for m in recent
    ])
    try:
        category_output = category_chain.invoke({
            "history": recent_history,
            "user_text": state["user_text"]
        })
        proposed_category = str(category_output).strip().lower()
        valid_categories = [c.lower() for c in all_categories]  # Updated to include 'opinion'
        proposed_category = proposed_category if proposed_category in valid_categories else "emotional_functioning"
        
        session_id = state["session_id"]
        if session_id not in session_category_counts:
            session_category_counts[session_id] = {}
        
        if proposed_category not in session_category_counts[session_id]:
            session_category_counts[session_id][proposed_category] = 1
        else:
            session_category_counts[session_id][proposed_category] += 1
        
        category_questions_count = session_category_counts[session_id][proposed_category]
        last_category = state.get("last_category", "")
            
    except Exception as e:
        logger.warning(f"Category classification failed: {e}. Defaulting to emotional_functioning.")
        proposed_category = "emotional_functioning"
        session_id = state["session_id"]
        if session_id not in session_category_counts:
            session_category_counts[session_id] = {}
        
        if proposed_category not in session_category_counts[session_id]:
            session_category_counts[session_id][proposed_category] = 1
        else:
            session_category_counts[session_id][proposed_category] += 1
        
        category_questions_count = session_category_counts[session_id][proposed_category]
        last_category = state.get("last_category", "")
    
    return {
        "category": proposed_category,
        "last_category": last_category,
        "category_questions_count": category_questions_count
    }

def score_emotion_node(state: GraphState) -> Dict[str, Any]:
    text = state["user_text"]
    try:
        scores = emotion_classifier(text)
        if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], list):
            scores = scores[0]
        emotion_dict = {item.get("label", "unknown"): round(item.get("score", 0.0), 4) for item in scores if isinstance(item, dict)}
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}")
        emotion_dict = {}
    return {"emotion_scores": emotion_dict}

def should_fetch_marks(state: GraphState) -> str:
    combined_text = str(state["user_text"]).lower() + " " + " ".join([str(m.content).lower() if hasattr(m, 'content') else "" for m in state["messages"][-5:]])
    school_keywords = ["school", "teacher", "marks", "exam", "study"]
    if any(keyword in combined_text for keyword in school_keywords):
        return "fetch"
    return "skip"

def fetch_marks_node(state: GraphState) -> Dict[str, Any]:
    name = state["name"] or "Unknown"
    try:
        marks_str = fetch_marks_direct(name)
        marks = marks_str if marks_str != "null" else None
        logger.info(f"Marks fetched: {marks}")
    except Exception as e:
        logger.warning(f"Marks fetch failed: {e}. Skipping.")
        marks = None
    return {"marks": marks}

def retrieve_rag_node(state: GraphState) -> Dict[str, Any]:
    category = state["category"]
    query = state["user_text"]
    contexts = retrieve_context(query, category)
    rag_context = "\n".join(contexts)
    return {"rag_context": rag_context}

def get_guideline_node(state: GraphState) -> Dict[str, Any]:
    cat = state["category"]
    guideline = get_guideline_prompt(cat)
    return {"guideline": guideline}



class AdaptedEmpathy(BaseModel):
    """JSON that the LLM must return."""
    adapted_response: str = Field(
        description="Empathetic response, rewritten for the current user & conversation context."
    )
    adapted_next_question: str = Field(
        description="Follow‑up question, rewritten for the current user & conversation context."
    )


def _build_adapt_prompt(
    sys_prompt: str,
    history_summary: str,
    user_text: str,
    base_empathic_q: str,
    base_empathic_rsp: str,
    base_next_q: str,
    pending_questions: Optional[List[str]] = None,
) -> str:
    """Return the full prompt that asks the LLM to rewrite the three template fields."""
    pending_str = "\n".join([f"- {q}" for q in (pending_questions or [])]) if pending_questions else ""
    pending_note = f"\n\nNote unanswered questions from history:\n{pending_str}" if pending_str else ""
    
    # Simple schema description (no full JSON dump to avoid echoing)
    schema_desc = """
{
  "adapted_response": "Your rewritten empathetic response (string, 1-50 words)",
  "adapted_next_question": "Your rewritten follow-up question (string, 1-30 words)"
}
    """
    
    # Add 1-2 examples to guide output
    examples = """
Examples of valid output (copy this exact format):

Input user: "I'm stressed about exams."
Template response: "Exams can be tough."
Template question: "How do you study?"

Output:
{{
  "adapted_response": "I hear how overwhelming exams feel right now—it's valid to feel stressed.",
  "adapted_next_question": "What part of studying feels most challenging for you?"
}}

Another example:
{{
  "adapted_response": "That sounds frustrating—it's okay to feel angry about unfair treatment.",
  "adapted_next_question": "What happened next in that situation?"
}}
    """
    
    return f"""
You are an empathetic mental-health chatbot.  
Below is a **template** for a generic user. Rewrite it for the **current conversation** (history + latest user message).

{pending_note}

--- USER HISTORY -------------------------------------------------
{history_summary}

--- USER LATEST MESSAGE -------------------------------------------
{user_text}

--- TEMPLATE EMPATHIC QUESTION ------------------------------------
{base_empathic_q}

--- TEMPLATE EMPATHIC RESPONSE ------------------------------------
{base_empathic_rsp}

--- TEMPLATE NEXT QUESTION ----------------------------------------
{base_next_q}

Rewrite using the history and user's feeling:
* Empathic response → Keep warm, validating, <50 words.
* Next question → Open-ended, <30 words. Reference pending questions if relevant.

Output **ONLY** the JSON data as a dictionary with keys 'adapted_response' and 'adapted_next_question'. 
Do **NOT** output the schema description, examples, explanations, or extra text. Start directly with {{.

Schema format:
{schema_desc}

{examples}
    """

def retrieve_best_template(user_text: str, category: Optional[str] = None, is_positive: bool = False) -> tuple[Document|None, List[Document]]:
    """
    Returns:
        - the chosen Document (or None if nothing matches)
        - the list of docs that were examined (for logging / citation)
    """
    vs = positive_vectorstore if is_positive else vectorstore
    if vs is None:
        return None, []

    # Retrieve a handful of candidates – we’ll keep the highest‑scoring one
    try:
        # Use FAISS similarity search directly to avoid retriever type issues
        candidates = vs.similarity_search(user_text, k=5)
    except Exception as e:
        logger.warning(f"Vectorstore similarity_search failed: {e}. Falling back to empty candidate list.")
        candidates = []

    # For positive sentiment, no category filtering
    # For negative, filter by category if provided
    if not is_positive and category:
        cat = (category or "").lower()
        filtered = [
            d for d in candidates
            if cat in d.metadata.get("category", "").lower()
        ]
        # If filtering kills everything, fall back to the original list
        if filtered:
            candidates = filtered

    best = candidates[0] if candidates else None
    return best, candidates


ADAPTED_EMPATHY_SCHEMA = json.dumps(
    AdaptedEmpathy.model_json_schema(),
    indent=2,
    ensure_ascii=False,
)

# -------------------------------------------------------
#  The *final* node – now robust against the dict‑vs‑object issue
# -------------------------------------------------------
# -------------------------------------------------------
#  The final node – fully typed and Pylance‑clean
# -------------------------------------------------------
def craft_response_node(state: GraphState) -> Dict[str, Any]:
    """
    1️⃣  Safety checks (self‑harm, moral‑risk, goodbye) – unchanged.
    2️⃣  Retrieve the best empathy template from FAISS.
    3️⃣  Build an adaptation prompt that contains the three template strings.
    4️⃣  Call the LLM, parse the JSON **into a Pydantic object**, and produce the final answer.
    5️⃣  Return both the answer and the source document (for citation).
    """
    # ------------------------------------------------------------------
    # 0️⃣  Safety / end‑chat pre‑filters (same as before)
    # ------------------------------------------------------------------
    

    if detect_moral_risk(state["user_text"]):
        safe_text = (
            "I hear that you’re feeling really upset right now. "
            "Wanting to hurt someone isn’t okay – you’re a kind person and there are better ways to deal with those feelings. "
            "Can you tell me what’s making you feel this way, or would you like to talk about a safer way to handle the situation?"
        )
        logger.info("Moral‑risk detected → safe reply.")
        return {"messages": [AIMessage(content=safe_text)], "source_documents": []}

    if detect_end_intent(state["user_text"]):
        goodbye_msg = (
            "It’s been wonderful chatting with you! You’re doing great, and remember you can always stay positive. "
            f"{ZEN_MODE_SUGGESTION} Take care and see you next time!"
        )
        logger.info("End‑intent detected → goodbye.")
        return {"messages": [AIMessage(content=goodbye_msg)], "source_documents": []}

    # ------------------------------------------------------------------
    # 1️⃣  History snapshot (max 6 turns, 30‑word each)
    # ------------------------------------------------------------------
    MAX_TURNS = 6
    recent = state["messages"][-MAX_TURNS:]
    history_lines: List[str] = []
    pending_questions = []
    for m in recent:
        prefix = "User:" if isinstance(m, HumanMessage) else "AI:"
        if isinstance(m, AIMessage) and "?" in (m.content if isinstance(m.content, str) else str(m.content)):
            pending_questions.append(str(m.content).strip())
        txt = " ".join(str(m.content).split()[:30])
        history_lines.append(f"{prefix} {txt}")
    history_summary = "\n".join(history_lines)

    # Trim pending questions to last few
    pending_questions = pending_questions[-3:]

    # Check self‑harm against a single string (combined recent history + latest user text)
    combined_check_text = history_summary
    if state.get("user_text"):
        combined_check_text = f"{history_summary}\nUser: {state['user_text']}"

    if detect_self_harm(combined_check_text):
        logger.warning("Self‑harm detected – returning crisis script.")
        return {"messages": [AIMessage(content=CRISIS_RESPONSE)], "source_documents": []}

    # ------------------------------------------------------------------
    # 2️⃣  Emotion tip (optional)
    # ------------------------------------------------------------------
    scores: Dict[str, float] = state.get("emotion_scores", {})
    if scores:
        # safest way – use items() → guarantees the key‑func returns a float
        top_emotion = max(scores.items(), key=lambda item: item[1])[0]  # type: ignore[arg-type]
        tip = EMOTION_TIPS.get(top_emotion, "")
    else:
        tip = ""

    # ------------------------------------------------------------------
    # 3️⃣  Retrieve the *best* empathy template from the FAISS store
    # ------------------------------------------------------------------
    template_doc: Optional[Document] = None
    source_documents: List[Document] = []       # will be returned for logging / UI

    sentiment = state.get("sentiment", "negative")
    category = state.get("category", None) if sentiment != "positive" else None
    is_positive = sentiment == "positive"

    vs = positive_vectorstore if is_positive else vectorstore
    if vs is not None:
        try:
            # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
            # Create a retriever that returns the top‑k most similar docs.
            # We ask for a few candidates (k=5) so we can optionally filter by category.

            retriever = cast(Any, vs.as_retriever(search_kwargs={"k": 5}))
            candidate_docs = retriever.invoke(state["user_text"])

            # For positive sentiment, skip category filtering to use any positive template.
            # For negative/default, filter by topic category.
            if is_positive:
                final_candidates = candidate_docs
            else:
                cat = (category or "").lower()
                filtered = [
                    d for d in candidate_docs
                    if cat in d.metadata.get("category", "").lower()
                ]
                final_candidates = filtered if filtered else candidate_docs
            logger.info(f"Positive retrieval: {len(candidate_docs)} candidates found for query '{state['user_text'][:50]}...' from {sentiment} vectorstore")
            # Take the very best match (first element after sorting by similarity).
            if final_candidates:
                template_doc = final_candidates[0]          # type: ignore[assignment]       # keep for citation / UI
                source_documents = final_candidates  # Optional: for logging/UI
                logger.info(f"Top candidate ID: {final_candidates[0].metadata.get('id')}")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    # ------------------------------------------------------------------
    # 4️⃣  If we have a template, **adapt** it with the LLM.
    # ------------------------------------------------------------------
    if template_doc is not None:
        # ------------------------------------------------------------------
        #   Pull the three raw template strings from metadata.
        # ------------------------------------------------------------------
        base_empathic_q   = template_doc.metadata.get("empathic_question", "")
        base_empathic_rsp = template_doc.metadata.get("empathic_response", "")
        base_next_q       = template_doc.metadata.get("next_question", "")
        sys_prompt = template_doc.metadata.get("system_prompt", SYSTEM_PROMPT)

        # ------------------------------------------------------------------
        #   Build the adaptation prompt.
        # ------------------------------------------------------------------
        adapt_prompt = _build_adapt_prompt(
            sys_prompt=sys_prompt,
            history_summary=history_summary,
            user_text=state["user_text"],
            base_empathic_q=base_empathic_q,
            base_empathic_rsp=base_empathic_rsp,
            base_next_q=base_next_q,
            pending_questions=pending_questions,
        )

        # ------------------------------------------------------------------
        #   Call the model and parse the JSON.
        # ------------------------------------------------------------------
        adapt_parser = JsonOutputParser(pydantic_object=AdaptedEmpathy)

        try:
            # The LLM returns an `AIMessage` (or sometimes a raw string).
            raw_msg = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=adapt_prompt)]
            )

            # ------------------------------------------------------------------
            #   Normalise the content to a plain string before feeding the parser.
            # ------------------------------------------------------------------
            if isinstance(raw_msg, AIMessage):
                content_obj = raw_msg.content
            else:
                content_obj = getattr(raw_msg, "content", raw_msg)   # fallback

            # The content may be a list of strings – join them.
            if isinstance(content_obj, list):
                content_str = "\n".join(str(p) for p in content_obj)
            else:
                content_str = str(content_obj)

            # Parse the JSON into a dict, then build the Pydantic model.
            parsed_dict = adapt_parser.parse(content_str)          # type: ignore[arg-type]
            adapted = AdaptedEmpathy(**parsed_dict)

            # ------------------------------------------------------------------
            #   Build the final answer that will be sent back to the user.
            # ------------------------------------------------------------------
            final_text = f"{adapted.adapted_response}\n\n{adapted.adapted_next_question}"
            # Optional citation line – mirrors the `ask_bot()` helper you showed.
            src_id = template_doc.metadata.get("id", "unknown")
            final_text = f"{final_text}\n\nSource: [{src_id}]"

        except Exception as exc:
            # ------------------------------------------------------------------
            #   Anything that goes wrong in the adaptation path falls back
            #   to a generic, safe reply (so the user never sees a crash).
            # ------------------------------------------------------------------
            logger.exception(f"Template adaptation failed: {exc}")
            final_text = (
                "I’m hearing you. It sounds like you’re feeling something important. "
                "Could you tell me a little more about what’s on your mind?"
            )
            source_documents = []   # no reliable source to cite

        # ------------------------------------------------------------------
        #   Return the adapted answer (and the doc we used as a source).
        # ------------------------------------------------------------------
        return {"messages": [AIMessage(content=final_text)], "source_documents": source_documents, "pending_questions": pending_questions}

    # ------------------------------------------------------------------
    # 5️⃣  **Fallback** – no matching template or retrieval failed.
    # ------------------------------------------------------------------
    logger.info("No empathy template found – falling back to the generic response chain.")

    # ---- Re‑use the generic USER_PROMPT you already have in the file ----
    # (You may still want to include the emotion tip, guideline, etc.)
    max_q_per_cat = state["max_questions"]
    cat_q_cnt = state.get("category_questions_count", 0)
    progress_pct = min(100, int((cat_q_cnt / max_q_per_cat) * 100) if max_q_per_cat else 0)

    # Re‑compute the tip (we already have it, but keep the logic self‑contained)
    scores: Dict[str, float] = state.get("emotion_scores", {})
    if scores:
        top_emotion = max(scores.items(), key=lambda item: item[1])[0]   # type: ignore[arg-type]
        tip = EMOTION_TIPS.get(top_emotion, "")
    else:
        tip = ""

    # Build the generic user prompt (identical to the one you used before)
    user_prompt = USER_PROMPT.format(
        max_turns=MAX_TURNS,
        history_summary=history_summary,
        user_text=state["user_text"],
        category=state.get("category", "emotional_functioning"),
        progress_pct=progress_pct,
        cat_q_cnt=cat_q_cnt,
        max_q_per_cat=max_q_per_cat,
        emotion_tip=tip,
        guideline=state.get("guideline", ""),
        probe_instruction=(
            "" if progress_pct > 5
            else "Do NOT draw any conclusions about the child's mental state yet; focus only on listening and validating."
        ),
        switch_instruction=(
            "" if cat_q_cnt < max_q_per_cat
            else "Ask the user whether they would like to elaborate more on this topic, switch to a related topic, or continue the current conversation."
        ),
    )

    # Generic chain that returns a ZenarkResponse (your original JSON schema)
    generic_chain = (ChatOpenAI(model="gpt-4o-mini", temperature=0.7) | response_parser)

    try:
        raw = generic_chain.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
        if isinstance(raw, ZenarkResponse):
            result = raw
        elif isinstance(raw, dict):
            result = ZenarkResponse(**raw)
        else:
            # raw is an AIMessage – parse its content with the same JSON parser
            result = ZenarkResponse(**response_parser.parse(raw.content))
        final_text = f"{result.validation} {result.reflection} {result.question}"
    except Exception as exc:
        logger.exception(f"Generic LLM failed: {exc}")
        final_text = "I’m here to listen. Could you tell me more about how you’re feeling?"

    # No source document for the generic path – return an empty list.
    return {"messages": [AIMessage(content=final_text)], "source_documents": [], "pending_questions": pending_questions}

def positive_craft_response(state: GraphState) -> Dict[str, Any]:
    """
    Specialized craft node for positive sentiment: skips category, suicide, etc.
    Directly retrieves from positive store and adapts.
    """
    # Reuse much of craft_response_node logic, but simplified
    MAX_TURNS = 6
    recent = state["messages"][-MAX_TURNS:]
    history_lines: List[str] = []
    pending_questions = []
    for m in recent:
        prefix = "User:" if isinstance(m, HumanMessage) else "AI:"
        if isinstance(m, AIMessage) and "?" in (m.content if isinstance(m.content, str) else str(m.content)):
            pending_questions.append(str(m.content).strip())
        txt = " ".join(str(m.content).split()[:30])
        history_lines.append(f"{prefix} {txt}")
    history_summary = "\n".join(history_lines)
    pending_questions = pending_questions[-3:]

    # Safety checks (moral risk, end intent)
    if detect_moral_risk(state["user_text"]):
        safe_text = (
            "I hear that you’re feeling really upset right now. "
            "Wanting to hurt someone isn’t okay – you’re a kind person and there are better ways to deal with those feelings. "
            "Can you tell me what’s making you feel this way, or would you like to talk about a safer way to handle the situation?"
        )
        logger.info("Moral‑risk detected → safe reply.")
        return {"messages": [AIMessage(content=safe_text)], "source_documents": []}

    if detect_end_intent(state["user_text"]):
        goodbye_msg = (
            "It’s been wonderful chatting with you! You’re doing great, and remember you can always stay positive. "
            f"{ZEN_MODE_SUGGESTION} Take care and see you next time!"
        )
        logger.info("End‑intent detected → goodbye.")
        return {"messages": [AIMessage(content=goodbye_msg)], "source_documents": []}

    template_doc: Optional[Document] = None
    source_documents: List[Document] = []

    if positive_vectorstore is not None:
        try:
            retriever = cast(Any, positive_vectorstore.as_retriever(search_kwargs={"k": 5}))
            candidate_docs = retriever.invoke(state["user_text"])
            final_candidates = candidate_docs  # No filtering for positive
            logger.info(f"Positive retrieval: {len(candidate_docs)} candidates found for query '{state['user_text'][:50]}...' from positive vectorstore")
            if final_candidates:
                template_doc = final_candidates[0]
                source_documents = final_candidates
                logger.info(f"Top positive candidate ID: {final_candidates[0].metadata.get('id')}")
        except Exception as e:
            logger.warning(f"Positive RAG retrieval failed: {e}")

    if template_doc is not None:
        base_empathic_q   = template_doc.metadata.get("empathic_question", "")
        base_empathic_rsp = template_doc.metadata.get("empathic_response", "")
        base_next_q       = template_doc.metadata.get("next_question", "")
        sys_prompt = template_doc.metadata.get("system_prompt", SYSTEM_PROMPT)

        adapt_prompt = _build_adapt_prompt(
            sys_prompt=sys_prompt,
            history_summary=history_summary,
            user_text=state["user_text"],
            base_empathic_q=base_empathic_q,
            base_empathic_rsp=base_empathic_rsp,
            base_next_q=base_next_q,
            pending_questions=pending_questions,
        )

        adapt_parser = JsonOutputParser(pydantic_object=AdaptedEmpathy)

        try:
            raw_msg = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).invoke(  # Lower temp for reliability
                [SystemMessage(content=sys_prompt), HumanMessage(content=adapt_prompt)]
            )

            if isinstance(raw_msg, AIMessage):
                content_obj = raw_msg.content
            else:
                content_obj = getattr(raw_msg, "content", raw_msg)

            if isinstance(content_obj, list):
                content_str = "\n".join(str(p) for p in content_obj)
            else:
                content_str = str(content_obj)

            logger.debug(f"Raw LLM response: {content_str[:300]}...")  # Increased for better debug

            # Regex to extract first JSON object (robust against wrappers like "Here is the JSON: {...}")
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None

            if isinstance(parsed, dict):
                # Check for schema indicators (expanded)
                schema_indicators = {'properties', 'required', 'title', '$defs', 'additionalProperties'}
                if any(key in parsed for key in schema_indicators) or 'description' in parsed.get('', {}):
                    logger.warning("LLM returned schema; falling back.")
                    raise ValueError("LLM returned schema instead of data")
                
                if "adapted_response" in parsed and "adapted_next_question" in parsed:
                    parsed_dict = parsed
                else:
                    # Let parser try the extracted JSON
                    parsed_dict = adapt_parser.parse(json_str)
            else:
                # Fallback to full content_str
                parsed_dict = adapt_parser.parse(content_str)

            # Final validation before Pydantic
            if not isinstance(parsed_dict, dict) or "adapted_response" not in parsed_dict or "adapted_next_question" not in parsed_dict:
                raise ValueError(f"Invalid response format after parsing: {type(parsed_dict)} - keys: {list(parsed_dict.keys()) if isinstance(parsed_dict, dict) else 'N/A'}")

            adapted = AdaptedEmpathy(**parsed_dict)

            final_text = f"{adapted.adapted_response}\n\n{adapted.adapted_next_question}"
            src_id = template_doc.metadata.get("id", "unknown")
            final_text = f"{final_text}\n\nSource: [{src_id}]"

        except Exception as exc:
            logger.exception(f"Positive template adaptation failed: {exc}")
            # Enhanced fallback with sentiment-aware response
            final_text = (
                "That's wonderful to hear—I'm glad things feel positive for you right now. "
                "What’s one small thing that’s bringing you joy today?"
            )
            source_documents = []

        return {"messages": [AIMessage(content=final_text)], "source_documents": source_documents, "pending_questions": pending_questions}

    # Fallback for positive
    logger.info("No positive empathy template found – falling back to generic positive response.")
    final_text = "Sounds like a great day! I'm happy for you. What's one thing you're looking forward to next?"
    return {"messages": [AIMessage(content=final_text)], "source_documents": [], "pending_questions": pending_questions}

       


# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("sentiment", sentiment_node)
workflow.add_node("suicide_check", suicide_check_node)
workflow.add_node("classify_category", classify_category)
workflow.add_node("score_emotion", score_emotion_node)
workflow.add_node("fetch_marks", fetch_marks_node)
workflow.add_node("retrieve_rag", retrieve_rag_node)
workflow.add_node("get_guideline", get_guideline_node)
workflow.add_node("craft_response", craft_response_node)
workflow.add_node("positive_craft_response", positive_craft_response)

workflow.set_entry_point("sentiment")
workflow.add_conditional_edges(
    "sentiment",
    lambda state: state["sentiment"],
    {"positive": "positive_craft_response", "negative": "suicide_check", "neutral": "suicide_check"}  # Neutral treated as negative for safety
)
workflow.add_conditional_edges(
    "suicide_check",
    lambda state: END if state.get("is_crisis", False) else "classify_category",
    {END: END, "classify_category": "classify_category"}
)
workflow.add_edge("classify_category", "score_emotion")
workflow.add_conditional_edges(
    "score_emotion",
    should_fetch_marks,
    {"fetch": "fetch_marks", "skip": "retrieve_rag"}
)
workflow.add_edge("fetch_marks", "retrieve_rag")
workflow.add_edge("retrieve_rag", "get_guideline")
workflow.add_edge("get_guideline", "craft_response")
workflow.add_edge("craft_response", END)
workflow.add_edge("positive_craft_response", END)

graph_app = workflow.compile()
logger.info("LangGraph workflow compiled successfully.")

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

# Store for session histories
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ============================================================
#  CORE RESPONSE LOGIC (LangGraph Integration)
# ============================================================
def generate_response(user_text: str, name: Optional[str] = None, question_index=1, max_questions=5, session_id: str = "default"):
    start_time = datetime.datetime.now()
    safe_name = name or "Unknown"
    logger.info(f"Generating response | user={safe_name} | session={session_id} | text='{user_text[:80]}'")

    if not user_text:
        return "Could you share a bit more about how you're feeling?"

    history = get_session_history(session_id)
    user_msg = HumanMessage(content=user_text)
    history.add_message(user_msg)

    initial_state = {
        "messages": list(history.messages),  # Copy to avoid mutation during invoke
        "user_text": user_text,
        "session_id": session_id,
        "name": name,
        "question_index": question_index,
        "max_questions": max_questions,
        "category": "emotional_functioning",
        "last_category": None,
        "category_questions_count": 0,
        "emotion_scores": {},
        "marks": None,
        "rag_context": "",
        "guideline": "",
        "sentiment": None,
        "is_crisis": False,
        "pending_questions": [],
    }

    start_invoke = datetime.datetime.now()
    result = graph_app.invoke(cast(GraphState, initial_state))
    elapsed_invoke = (datetime.datetime.now() - start_invoke).total_seconds()

    # Sync history with result
    history.messages = result["messages"]

    response = result["messages"][-1].content
    category = result.get("category", "unknown")
    cat_count = result.get("category_questions_count", 0)
    progress = min(100, int((cat_count / max_questions) * 100))
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Response generated | category={category} | cat_count={cat_count} | progress={progress}% | invoke_time={elapsed_invoke:.2f}s | total_time={elapsed:.2f}s | len={len(response)} chars")
    return response

def save_conversation(conversation, user_name: Optional[str], session_id: str = "default"):
    logger.info(f"Saving conversation for user={user_name} | session={session_id}")
    analyzed_conversation = []
    for turn in conversation:
        if "user" in turn:
            text = turn["user"]
            try:
                scores = emotion_classifier(text)
                if isinstance(scores, list) and isinstance(scores[0], list):
                    scores = scores[0]
                emotion_dict = {item.get("label", "unknown"): round(item.get("score", 0.0), 4) for item in scores if isinstance(item, dict)}
                turn["emotion_scores"] = emotion_dict
            except Exception as e:
                logger.warning(f"Emotion classification failed: {e}")
                turn["emotion_scores"] = {"error": str(e)}
        analyzed_conversation.append(turn)

    record = {
        "name": user_name or "Unknown",
        "conversation": analyzed_conversation,
        "timestamp": datetime.datetime.now(),
    }

    try:
        result = chats_col.insert_one(record)
        record["_id"] = result.inserted_id
    except Exception as e:
        logger.exception(f"Mongo insert failed: {e}")

    folder = "chat_sessions"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{user_name or 'unknown'}_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=safe_json)
    logger.info(f"Conversation persisted to {path}")
    return convert_objectid(record)  # Safe for JSON


def generate_report(name: str):
    record = db["chat_sessions"].find_one({"name": name}, sort=[("_id", -1)])
    if not record:
        return {"error": f"No conversation found for {name}"}
    conv_text = "\n".join(f"User: {t.get('user','')}\nAI: {t.get('ai','')}" for t in record.get("conversation", []))
    report_data = generate_autogen_report(conv_text, name)
    reports_col.insert_one(report_data)
    return report_data

# # ============================================================
# #  SCHEMAS
# # ============================================================
# class ChatRequest(BaseModel):
#     text: str
#     name: Optional[str] = None
#     question_index: int = 1
#     max_questions: int = 5  # Updated default to 5
#     session_id: str = "default"  # New: for per-session memory

# class SaveRequest(BaseModel):
#     conversation: List[Dict]
#     name: Optional[str] = None
#     session_id: str = "default"


# class ReportRequest(BaseModel):
#     name: str


# ============================================================
#  MIDDLEWARE & ENDPOINTS
# ============================================================
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start = datetime.datetime.now()
#     logger.info(f"Request start | {request.method} {request.url.path}")
#     try:
#         response = await call_next(request)
#     except Exception as e:
#         logger.exception(f"Unhandled error during {request.url.path}: {e}")
#         return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
#     duration = (datetime.datetime.now() - start).total_seconds()
#     logger.info(f"Request end | {request.method} {request.url.path} | {duration:.2f}s | Status={response.status_code}")
#     return response

# @app.get("/health")
# def health_check():
#     logger.info("Health check pinged.")
#     return {"status": "ok", "time": datetime.datetime.now().isoformat()}

# @app.post("/chat")
# def chat_endpoint(req: ChatRequest):
#     safe_name = req.name or "Unknown"  # Handle None for logging
#     response = generate_response(req.text, req.name, req.question_index, req.max_questions, req.session_id)
#     logger.info(f"Chat response returned to user={safe_name} | session={req.session_id}")
#     return {"response": response}

# @app.post("/save_chat")
# def save_chat_endpoint(req: SaveRequest):
#     record = save_conversation(req.conversation, req.name, req.session_id)
#     return JSONResponse(content=jsonable_encoder(record))


# @app.post("/generate_report")
# def generate_report_endpoint(req: ReportRequest):
#     """Generate and store a multi-agent reflective report from user's latest conversation."""
#     try:
#         record = db["chat_sessions"].find_one({"name": req.name}, sort=[("_id", -1)])
#         if not record or "conversation" not in record:
#             return JSONResponse(status_code=404, content={"error": f"No conversation found for {req.name}"})

#         conv_text = []
#         for t in record["conversation"]:
#             user_msg = t.get("user", "")
#             ai_msg = t.get("ai", "")
#             if user_msg or ai_msg:
#                 conv_text.append(f"User: {user_msg}\nAI: {ai_msg}")
#         conv_text = "\n".join(conv_text)

#         report_data = generate_autogen_report(conv_text, req.name)
#         insert_result = reports_col.insert_one(report_data)
#         report_data["_id"] = insert_result.inserted_id

#         # Convert ObjectId -> str before returning
#         safe_report = convert_objectid(report_data)
#         return JSONResponse(content=safe_report)

#     except Exception as e:
#         logger.exception(f"Report generation failed: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================
#  MAIN ENTRYPOINT
# ============================================================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_config=None)