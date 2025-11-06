# app.py
import os, re, json, torch, datetime, logging, numpy as np
from typing import Optional, List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import datetime
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    pipeline
)
from huggingface_hub import login
from bson import ObjectId
from langchain.tools import tool
from langchain_classic.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.agents import create_agent
from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from logging.handlers import RotatingFileHandler

# ============================================================
#  LOGGING CONFIGURATION
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

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

import sys
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

logging.basicConfig(encoding='utf-8', level=logging.INFO)


logger = logging.getLogger("zenark")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================
#  ENVIRONMENT + INITIALIZATION
# ============================================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



logger.info("Server startup initiated.")
logger.info(f"Environment loaded. HF_TOKEN={'set' if HF_TOKEN else 'missing'} | MONGO_URI={'set' if MONGO_URI else 'missing'}")

# ============================================================
#  DATABASE CONNECTION
# ============================================================
try:
    client = MongoClient(MONGO_URI)
    db = client["zenark_db"]
    marks_col = db["student_marks"]
    chats_col = db["chat_sessions"]
    logger.info("✅ MongoDB connection established.")
except Exception as e:
    logger.exception(f"MongoDB connection failed: {e}")
    raise

# ============================================================
#  HUGGINGFACE LOGIN
# ============================================================
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("✅ Logged into Hugging Face Hub successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Hugging Face login failed: {e}")
else:
    logger.warning("⚠️ No HUGGINGFACE_TOKEN found in environment.")

# ============================================================
#  MODEL LOADING
# ============================================================
# MODEL_NAME = "arnir0/Tiny-LLM"
# logger.info(f"Loading base model: {MODEL_NAME}")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

chat_history = ChatMessageHistory()

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["category"] = record.get("category")
    metadata["system_prompt"] = record.get("system_prompt")
    return metadata

def content_builder(record: dict) -> str:
    """Combine key empathy fields into a unified text block."""
    q = record.get("empathic_question", "")
    r = record.get("empathic_response", "")
    n = record.get("next_question", "")
    return f"Empathic Question: {q}\nEmpathic Response: {r}\nNext Question: {n}"

logger.info("Loading embedding models and emotion classifier...")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    token=HF_TOKEN,
    top_k=None,
)
logger.info("All models successfully initialized.")




def convert_objectid(doc):
    """Recursively converts ObjectId to string for nested dicts/lists."""
    if isinstance(doc, list):
        return [convert_objectid(i) for i in doc]
    elif isinstance(doc, dict):
        return {k: convert_objectid(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

# ============================================================
#  LOAD RAG DATASET
# ============================================================
DATA_PATH = "combined_dataset.json"

# Load JSON correctly — note the path ".dataset[]"
loader = JSONLoader(
    file_path="combined_dataset.json",
    jq_schema=".dataset[]",
    content_key="empathic_question",  # temporary
    metadata_func=metadata_func,
)

raw_docs = loader.load()


# Rebuild composite empathy text
docs = []
for d in raw_docs:
    record = d.metadata
    q = record.get("empathic_question", "")
    r = record.get("empathic_response", "")
    n = record.get("next_question", "")
    combined_text = f"Empathic Question: {q}\nEmpathic Response: {r}\nNext Question: {n}"
    docs.append(Document(page_content=combined_text, metadata=record))

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embedding_function)
vectorstore.save_local("faiss_index")

print(f"✅ Loaded {len(docs)} empathy records successfully.")

# raw_docs = loader.load()

# # Convert manually to LangChain Documents
# from langchain_core.documents import Document

# docs = [
#     Document(
#         page_content=f"{d.metadata.get('empathic_question', '')} {d.metadata.get('empathic_response', '')}",
#         metadata={
#             "category": d.metadata.get("category", ""),
#             "system_prompt": d.metadata.get("system_prompt", "")
#         }
#     )
#     for d in raw_docs
# ]

docs = loader.load()
print(f"Loaded {len(docs)} empathy documents.")
print(docs[0].page_content)
print(docs[0].metadata)


# ============================================================
#  LANGCHAIN CORE
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_function)
vectorstore.save_local("faiss_zenark_index")
print("✅ FAISS index built and saved with metadata.")
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

summary_memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="summary",           # ✅ Give it a unique key
    return_messages=True
)

combined_memory = CombinedMemory(
    memories=[buffer_memory, summary_memory]            # ✅ Force consistent input
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=combined_memory
)

# ------------------------------------------------------------
# Helper — Extract categories dynamically from dataset
# ------------------------------------------------------------
def extract_categories_from_json(file_path: str):
    """Extract all unique categories from a JSON dataset file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("dataset", data)
    categories = sorted({item.get("category", "").strip() for item in records if "category" in item})
    return categories

# ------------------------------------------------------------
# Tool 1 — Conversation history (context)
# ------------------------------------------------------------
@tool
def get_conversation_history(history: str) -> str:
    """Return the full conversation history for reasoning."""
    return f"Conversation history:\n{history}"

# ------------------------------------------------------------
# Tool 2 — Fetch academic marks from MongoDB
# ------------------------------------------------------------
@tool
def get_academic_marks(student_name: str) -> str:
    """Retrieve academic marks from MongoDB for the given student."""
    record = marks_col.find_one({"name": {"$regex": f"^{student_name}$", "$options": "i"}})
    if not record:
        return f"No marks found for {student_name}."

    marks_list = record.get("marks", [])
    formatted = ", ".join(f"{m['subject']}: {m['marks']}" for m in marks_list)
    return f"Academic report for {student_name}: {formatted}"

categories = extract_categories_from_json("combined_dataset.json")

# ------------------------------------------------------------
# Category + Academic Retrieval Agent
# ------------------------------------------------------------
category_agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_conversation_history, get_academic_marks],
    system_prompt=(
        f"You are a clinical assistant analyzing adolescent conversations.\n"
        f"Use the `get_conversation_history` tool to review context.\n"
        f"Classify the conversation into one of the categories:\n"
        f"{categories}.\n"
        f"If the topic mentions school, teachers, study, or exams, also call `get_academic_marks` "
        f"to fetch the student's marks from the database using their name.\n"
        f"Output the result as JSON:\n"
        f"{{'category': '<category>', 'marks': '<marks or None>'}}"
    ),
)

# ============================================================
#  IMPORT GUIDELINES
# ============================================================
from Guideliness import (
    action_cognitive_guidelines,
    action_emotional_guidelines,
    action_developmental_guidelines,
    action_environmental_guidelines,
    action_family_guidelines,
    action_medical_guidelines,
    action_peer_guidelines,
    action_school_guidelines,
    action_social_support_guidelines,
)

GUIDELINES = {
    "family_issues": f"{action_family_guidelines}",
    "school_stress": f"{action_school_guidelines}",
    "peer_issues": f"{action_peer_guidelines}",
    "developmental_details": f"{action_developmental_guidelines}",
    "medical_complaint": f"{action_medical_guidelines}",
    "emotional_distress": f"{action_emotional_guidelines}",
    "environmental_stress": f"{action_environmental_guidelines}",
    "cognitive_concern": f"{action_cognitive_guidelines}",
    "social_support": f"{action_social_support_guidelines}",
}

# Complete mapping for all your dataset categories
CATEGORY_MAP = {
    # Map exact category names from your dataset to guideline keys
    "anxiety": "emotional_distress",
    "bipolar": "emotional_distress", 
    "depression": "emotional_distress",
    "emotional_functioning": "emotional_distress",
    "environmental_stressors": "environmental_stress",
    "family": "family_issues",  # ✅ Fixed: was "family_conflict"
    "ocd": "emotional_distress",
    "peer_relations": "peer_issues",  # ✅ This one was correct
    "ptsd": "emotional_distress",
    
    # Keep your existing mappings for backward compatibility
    "school_stress": "school",
    "developmental": "developmental",
    "medical": "medical",
    "emotional_distress": "emotional",
    "environmental": "environmental",
    "cognitive": "cognitive",
    "social_support": "support",
}



CORPUS = [
    f"{getattr(d, 'metadata', {}).get('category', '')} | "
    f"{getattr(d, 'metadata', {}).get('system_prompt', '')} "
    f"{getattr(d, 'page_content', '')}"
    for d in docs
]



CORPUS_EMB = torch.tensor(
    np.array([embedding_function.embed_query(text) for text in CORPUS])
)
logger.info("Corpus embeddings prepared.")

# ============================================================
#  HELPERS
# ============================================================
def safe_json(o):
    if isinstance(o, (datetime.datetime,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="ignore")
    return str(o)


def retrieve_context(query, top_k=3, threshold=0.25):
    # Get embedding and ensure it's a NumPy array
    q_emb = np.array(embedding_function.embed_query(query)).reshape(1, -1)

    # Compute cosine similarity
    corpus_matrix = (
        CORPUS_EMB.numpy() if hasattr(CORPUS_EMB, "numpy") else np.array(CORPUS_EMB)
    )
    sims = cosine_similarity(q_emb, corpus_matrix)[0]

    idxs = sims.argsort()[-top_k:][::-1]
    results = [(CORPUS[i], sims[i]) for i in idxs if sims[i] >= threshold]
    return [r[0] for r in results]

def generate_fallback_response(prompt: str) -> str:
    """Generate response directly using LLM when RAG fails. Always returns a string."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])

        # Normalize all possible return types to string
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return str(response.content)
        if isinstance(response, list):
            return " ".join(str(getattr(r, "content", r)) for r in response)
        return str(response)

    except Exception as e:
        logger.error(f"Fallback response also failed: {e}")
        return (
            "I hear this is really challenging for you. Would you like to talk more about what's been going on?"
        )


# ============================================================
#  CORE RESPONSE LOGIC - Fixed Import Version
# ============================================================
def generate_response(user_text: str, name: Optional[str] = None, question_index=1, max_questions=10):
    start_time = datetime.datetime.now()
    logger.info(f"Generating response | user={name or 'Unknown'} | text='{user_text[:80]}'")

    if not user_text:
        return "Could you share a bit more about how you're feeling?"

    chat_history.add_user_message(user_text)

    # ------------------------------------------------------------
    # Step 1: Simple classification
    # ------------------------------------------------------------
    text_lower = user_text.lower()
    if any(word in text_lower for word in ["parent", "mom", "dad", "family"]):
        category = "family"
    elif any(word in text_lower for word in ["friend", "peer", "buddy"]):
        category = "peer_relations"
    elif any(word in text_lower for word in ["school", "teacher", "exam"]):
        category = "environmental_stressors"
    else:
        category = "emotional_functioning"

    # ------------------------------------------------------------
    # Step 2: Build natural prompt
    # ------------------------------------------------------------
    history_text = "\n".join(
        [f"User: {m.content}" if m.type == "human" else f"You: {m.content}" for m in chat_history.messages[-4:]]
    )
    
    progress = int((question_index / max_questions) * 100)
    
    prompt = f"""
You're having a warm, natural conversation with a teenager who's opening up about personal struggles.

Context: They're discussing {category} issues. 

Recent conversation:
{history_text}

They just said: "{user_text}"

Respond naturally like a caring adult who's genuinely interested:
- Show empathy and understanding
- Ask open-ended questions to learn more  
- Connect with their emotions
- Keep it conversational, not clinical
- Sound like a real person, not a robot

Respond in a warm, natural way:
"""
    
    logger.info(f"Natural conversation prompt:\n{prompt}")

    # ------------------------------------------------------------
    # Step 3: Generate response (FIXED IMPORT)
    # ------------------------------------------------------------
    try:
        # Try the most common import patterns
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
            response_obj = llm.invoke([HumanMessage(content=prompt)])
            response = response_obj.content
        except ImportError:
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
                response_obj = llm.invoke([HumanMessage(content=prompt)])
                response = response_obj.content
            except ImportError:
                # Fallback to direct OpenAI API
                import openai
                client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                ).choices[0].message.content
        
    except Exception as e:
        logger.exception(f"LLM error: {e}")
        response = "That sounds really challenging. I'm here to listen if you want to share more."

       # ------------------------------------------------------------
        # Step 4: Store and return
        # ------------------------------------------------------------
        # Normalize response to a guaranteed string
        if isinstance(response, list):
            response = " ".join(str(getattr(r, "content", r)) for r in response)
        elif response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(getattr(response, "content", response))

        chat_history.add_ai_message(response)
        elapsed = (datetime.datetime.now() - start_time).total_seconds()

        logger.info(f"Response generated | category={category} | progress={progress}% | time={elapsed:.2f}s")
        return response.strip()


def save_conversation(conversation, user_name: Optional[str]):
    logger.info(f"Saving conversation for user={user_name}")
    analyzed_conversation = []
    for turn in conversation:
        if "user" in turn:
            text = turn["user"]
            try:
                scores = emotion_classifier(text)
                # Flatten the list if nested
                if isinstance(scores, list) and isinstance(scores[0], list):
                    scores = scores[0]
                emotion_dict = {}
                for item in scores:
                    if isinstance(item, dict):
                        label = str(item.get("label", "unknown"))
                        score = round(float(item.get("score", 0.0)), 4)
                        emotion_dict[label] = score
                turn["emotion_scores"] = emotion_dict

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
        chats_col.insert_one(record)
    except Exception as e:
        logger.exception(f"Mongo insert failed: {e}")

    folder = "chat_sessions"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{user_name}_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=safe_json)
    logger.info(f"Conversation persisted to {path}")
    return record

# ============================================================
#  FASTAPI APP
# ============================================================
app = FastAPI(title="Zenark Mental Health API", version="2.0", description="Empathetic AI counseling system with detailed logging.")

class ChatRequest(BaseModel):
    text: str
    name: Optional[str] = None
    question_index: int = 1
    max_questions: int = 10

class SaveRequest(BaseModel):
    conversation: List[Dict]
    name: Optional[str] = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.datetime.now()
    logger.info(f"Request start | {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled error during {request.url.path}: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    duration = (datetime.datetime.now() - start).total_seconds()
    logger.info(f"Request end | {request.method} {request.url.path} | {duration:.2f}s | Status={response.status_code}")
    return response

@app.get("/health")
def health_check():
    logger.info("Health check pinged.")
    return {"status": "ok", "time": datetime.datetime.now().isoformat()}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    response = generate_response(req.text, req.name, req.question_index, req.max_questions)
    logger.info(f"Chat response returned to user={req.name}")
    return {"response": response}

@app.post("/save_chat")
async def save_chat_endpoint(request: Request):
    data = await request.json()
    
    # use your defined Mongo collection
    result = chats_col.insert_one(data)

    # Prepare response
    saved_doc = {**data, "_id": result.inserted_id}
    safe_doc = convert_objectid(saved_doc)

    return JSONResponse(content=jsonable_encoder(safe_doc))
# Run with:
# uvicorn app:app --reload
