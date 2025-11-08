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
    logger.info("All models successfully initialized.")
except Exception as e:
    logger.exception(f"Model loading failed: {e}")
    raise

# Your functions (unchanged)
def metadata_func(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata["id"] = record.get("id")
    metadata["category"] = record.get("category")
    metadata["system_prompt"] = record.get("system_prompt")
    return metadata

def content_builder(record: Dict[str, Any]) -> str:
    """Combine key empathy fields into a unified text block."""
    q = record.get("empathic_question", "")
    r = record.get("empathic_response", "")
    n = record.get("next_question", "")
    return f"Empathic Question: {q}\nEmpathic Response: {r}\nNext Question: {n}"

# ============================================================
#  LOAD RAG DATASET (Fully Fixed)
# ============================================================
DATA_PATH = "combined_dataset.json"
try:
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema=".dataset[]",  # Matches your {"dataset": [...]} structure
        content_key=None,  # Full record as dict
        metadata_func=metadata_func,
        text_content=False,  # Allow dict for page_content
    )
    raw_docs = loader.load()
    logger.info(f"Loaded {len(raw_docs)} raw empathy documents.")

    # Post-process: Convert dict page_content to string via content_builder
    docs = []
    for doc in raw_docs:
        try:
            raw_content = doc.page_content
            if isinstance(raw_content, str):
                # Fallback: Parse str to dict if loader quirked
                try:
                    record = json.loads(raw_content)
                except json.JSONDecodeError:
                    logger.warning(f"Unparsable str content: {raw_content[:50]}... Using raw str.")
                    doc.page_content = raw_content  # Keep as-is
                    docs.append(doc)
                    continue
            else:
                # Expected: dict from your structure
                record = cast(Dict[str, Any], raw_content)  # Type assertion (no runtime cost)
            
            # Build string from fields (e.g., "Empathic Question: ...")
            doc.page_content = content_builder(record)
            docs.append(doc)
        except Exception as e:
            logger.warning(f"Failed to process doc {doc.metadata.get('id', 'unknown')}: {e}. Skipping.")
            continue

    if docs:
        logger.info(f"Successfully processed {len(docs)} documents into string content.")
        # Debug: Sample output (remove in prod)
        print(f"Sample page_content: {docs[0].page_content[:100]}...")  # Truncated
        print(f"Sample metadata: {docs[0].metadata}")
    else:
        logger.error("No valid documents processed—check dataset structure or jq_schema.")

    # Build vectorstore (only if docs exist)
    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_function)
        vectorstore.save_local("faiss_zenark_index")
        logger.info("FAISS index built and saved with metadata.")

        # Corpus for similarity retrieval (category-prefixed for better matching)
        CORPUS = [
            f"{d.metadata.get('category', 'general')} | "
            f"{d.metadata.get('system_prompt', '')} "
            f"{d.page_content}"
            for d in docs
        ]
        CORPUS_EMB = torch.tensor(np.array([embedding_function.embed_query(text) for text in CORPUS]))
        logger.info("Corpus embeddings prepared.")
    else:
        vectorstore = None
        CORPUS = []
        CORPUS_EMB = torch.tensor([])
        logger.warning("No corpus built—RAG will use direct LLM fallback.")

except Exception as e:
    logger.exception(f"Dataset loading failed: {e}")
    docs = []
    vectorstore = None
    CORPUS = []
    CORPUS_EMB = torch.tensor([])



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


CRISIS_RESPONSE = (
    "I’m really sorry you’re feeling this way. It sounds like you’re thinking about ending your life, and that’s a serious sign you deserve help right now. "
    "Please call the National Suicide Prevention Helpline +91 9152987821 (India) or 988 (US) immediately. "
    "You can also text “HELLO” to 741741 (Crisis Text Line, US) – we’ll route you to an Indian service if needed, or call 112 (or 911) or go to the nearest emergency department. "
    "Would you like me to give you any other resources for your location?"
)


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
    cat_corpus = [c for c in CORPUS if category.lower() in c.lower()]
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



def craft_response_node(state: GraphState) -> Dict[str, Any]:
    # --------------------------------------------------------------
    # 1️⃣  History snapshot (max 6 turns, 30‑word each)
    # --------------------------------------------------------------
    MAX_TURNS = 6
    recent = state["messages"][-MAX_TURNS:]
    history_lines = []
    for m in recent:
        prefix = "User:" if isinstance(m, HumanMessage) else "AI:"
        txt = " ".join(str(m.content).split()[:30])   # truncate to ~30 words
        history_lines.append(f"{prefix} {txt}")
    history_summary = "\n".join(history_lines)

    # --------------------------------------------------------------
    # 2️⃣  Numeric helpers
    # --------------------------------------------------------------
    max_q_per_cat = state["max_questions"]               # usually 5
    cat_q_cnt = state["category_questions_count"]
    progress_pct = min(100, int((cat_q_cnt / max_q_per_cat) * 100))

    # --------------------------------------------------------------
    # 3️⃣  Flags for business rules
    # --------------------------------------------------------------
    can_probe      = progress_pct > 5                     # rule‑1
    need_switch_q = cat_q_cnt >= max_q_per_cat            # rule‑2

    # --------------------------------------------------------------
    # 4️⃣  Emotion tip (if any)
    # --------------------------------------------------------------
    scores: Dict[str, float] = state["emotion_scores"]
    if scores:
        top_emotion: str = max(scores, key=lambda k: scores[k])
        tip: str = EMOTION_TIPS.get(top_emotion, "")
    else:
        tip = ""

    # --------------------------------------------------------------
    # 5️⃣  **Self‑harm / suicidal‑ideation pre‑filter** (priority #1)
    # --------------------------------------------------------------
    if detect_self_harm(state["user_text"]):
        logger.warning("Self‑harm detected – returning crisis script.")
        return {"messages": [AIMessage(content=CRISIS_RESPONSE)]}

    # --------------------------------------------------------------
    # 6️⃣  **Moral‑risk pre‑filter** (violent / illegal intent) (priority #2)
    # --------------------------------------------------------------
    if detect_moral_risk(state["user_text"]):
        moral_validation = "I hear that you’re feeling really upset right now."
        moral_reflection = (
            "Wanting to hurt someone isn’t okay – you’re a kind person and there are better ways to deal with those feelings."
        )
        moral_question = (
            "Can you tell me what’s making you feel this way, or would you like to talk about a safer way to handle the situation?"
        )
        safe_text = f"{moral_validation} {moral_reflection} {moral_question}"
        logger.info("Moral‑risk detected → returning safe handcrafted reply.")
        return {"messages": [AIMessage(content=safe_text)]}

    # --------------------------------------------------------------
    # 7️⃣  **End‑chat pre‑filter** (priority #3)
    # --------------------------------------------------------------
    if detect_end_intent(state["user_text"]):
        goodbye_msg = (
            "It’s been wonderful chatting with you! You’re doing great, and remember you can always stay positive. "
            f"{ZEN_MODE_SUGGESTION} Take care and see you next time!"
        )
        logger.info("End‑intent detected → returning warm goodbye.")
        return {"messages": [AIMessage(content=goodbye_msg)]}

    # --------------------------------------------------------------
    # 8️⃣  **Progress < 5 %** → simple rapport‑only response (priority #4)
    # --------------------------------------------------------------
    if not can_probe:
        fallback = (
            "I hear you. It sounds like you’re dealing with something important. "
            "What’s on your mind right now?"
        )
        logger.info("Progress < 5 % → returning built‑in rapport response.")
        return {"messages": [AIMessage(content=fallback)]}

    # --------------------------------------------------------------
    # 9️⃣  **Question limit reached** → ask to continue or switch (priority #5)
    # --------------------------------------------------------------
    if need_switch_q:
        switch_prompt = (
            "We’ve talked a lot about this topic already. "
            "Would you like to explore it a bit more, or would you prefer to talk about something else?"
        )
        logger.info("Question limit reached → returning switch‑topic prompt.")
        return {"messages": [AIMessage(content=switch_prompt)]}

    # --------------------------------------------------------------
    # 🔟  (Optional) tiny flag‑in‑prompt strings for the LLM
    # --------------------------------------------------------------
    probe_instruction = (
        "" if can_probe else
        "Do NOT draw any conclusions about the child's mental state yet; focus only on listening and validating."
    )
    switch_instruction = (
        "" if not need_switch_q else
        "Ask the user whether they would like to elaborate more on this topic, "
        "switch to a related topic, or continue the current conversation."
    )

    # --------------------------------------------------------------
    # 1️⃣1️⃣  Assemble the messages that go to the LLM
    # --------------------------------------------------------------
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT.format(
                max_turns=MAX_TURNS,
                history_summary=history_summary,
                user_text=state["user_text"],
                category=state["category"],
                progress_pct=progress_pct,
                cat_q_cnt=cat_q_cnt,
                max_q_per_cat=max_q_per_cat,
                emotion_tip=tip,
                guideline=state["guideline"],
                probe_instruction=probe_instruction,
                switch_instruction=switch_instruction,
            )
        ),
    ]

    # --------------------------------------------------------------
    # 1️⃣2️⃣  Call the model → JSON parser → ZenarkResponse
    # --------------------------------------------------------------
    try:
        chain = (ChatOpenAI(model="gpt-4o-mini", temperature=0.7) | response_parser)

        raw = chain.invoke(messages)               # could be dict or ZenarkResponse
        if isinstance(raw, ZenarkResponse):
            result = raw
        elif isinstance(raw, dict):
            result = ZenarkResponse(**raw)
        elif hasattr(raw, "content"):
            parsed = response_parser.parse(raw.content)   # returns dict
            result = ZenarkResponse(**parsed)
        else:
            raise RuntimeError(f"Unexpected LLM output type: {type(raw)}")

        final_text = f"{result.validation} {result.reflection} {result.question}"
        logger.debug(f"Structured response: {result.model_dump_json()}")
    except Exception as exc:
        logger.exception(f"LLM failed, falling back – {exc}")
        final_text = (
            "I’m here to listen. Could you tell me more about how you’re feeling?"
        )

    # --------------------------------------------------------------
    # 1️⃣3️⃣  Return updated chat history
    # --------------------------------------------------------------
    return {"messages": [AIMessage(content=final_text)]}



# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("classify_category", classify_category)
workflow.add_node("score_emotion", score_emotion_node)
workflow.add_node("fetch_marks", fetch_marks_node)
workflow.add_node("retrieve_rag", retrieve_rag_node)
workflow.add_node("get_guideline", get_guideline_node)
workflow.add_node("craft_response", craft_response_node)

workflow.set_entry_point("classify_category")
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
    }

    start_invoke = datetime.datetime.now()
    result = graph_app.invoke(cast(GraphState, initial_state))
    elapsed_invoke = (datetime.datetime.now() - start_invoke).total_seconds()

    # Sync history with result
    history.messages = result["messages"]

    response = result["messages"][-1].content
    category = result["category"]
    cat_count = result["category_questions_count"]
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