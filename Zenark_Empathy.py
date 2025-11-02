#!pip install langchain langchain-core langchain-community==0.3.22 langchain-openai==0.2.0 faiss-cpu

# ============================================================
#  Child Adaptive RAG Chatbot with LangChain Memory Integration
# ============================================================

import re, random, datetime, torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from openai import OpenAI
import os
import json, os, datetime
# LangChain imports
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from pymongo import MongoClient
from typing import Optional
import numpy as np

from dotenv import load_dotenv
load_dotenv()

llm =  ChatOpenAI(model="gpt-4o-mini",temperature=0.5)


# ─────────────────────────────
#  MONGODB CONNECTION
# ─────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["zenark_db"]
marks_col = db["student_marks"]
chats_col = db["chat_sessions"]
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Logged into Hugging Face Hub successfully.")
    except Exception as e:
        print(f"⚠️ Hugging Face login failed: {e}")
else:
    print("⚠️ No HUGGINGFACE_TOKEN found in environment.")


# ─────────────────────────────
#  LOAD MODELS
# ─────────────────────────────
embedder_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN
)
embedder_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN
)

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    token=HF_TOKEN,
    return_all_scores=True
)

# ─────────────────────────────
#  CONTEXT AREAS (from clinical guidelines)
# ─────────────────────────────
CONTEXT_AREAS = {
    "family": [
        "father", "mother", "parents", "siblings", "home",
        "caretaker", "discipline", "attachment", "conflict"
    ],
    "school": [
        "teacher", "exam", "homework", "study", "grades", "learning",
        "attention", "classmates", "bullying", "school refusal"
    ],
    "peer_relations": [
        "friends", "best friend", "play", "isolation", "group",
        "social media", "peer pressure", "trust", "rejection"
    ],
    "developmental_history": [
        "speech", "language", "toilet training", "temper tantrums",
        "sleep patterns", "feeding habits", "milestones"
    ],
    "medical_physical": [
        "illness", "pain", "fatigue", "sleep", "headache", "stomach",
        "appetite", "energy", "body image"
    ],
    "emotional_functioning": [
        "sad", "fear", "anger", "mood", "worry", "hopeless", "irritable",
        "crying", "self-esteem"
    ],
    "environmental_stressors": [
        "neighbour", "violence", "financial", "housing", "community",
        "trauma", "migration", "loss"
    ],
    "cognitive_behavioral": [
        "thinking", "focus", "memory", "concentration",
        "obsession", "compulsion", "intrusive thoughts"
    ],
    "social_support": [
        "friends", "relatives", "counsellor", "teacher support",
        "safe space", "trust", "guidance"
    ]
}


def safe_json(o):
    """Convert non-serializable types to strings or lists."""
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="ignore")
    return str(o)


def get_user_marks(name: str | None, marks_col):
    """Fetch and format a student's marks from MongoDB."""
    record = marks_col.find_one({"name": name})
    if not record or not record.get("marks"):
        return None
    marks_text = "\n".join([f"{m['subject']}: {m['marks']}/100" for m in record["marks"]])
    avg = sum(m["marks"] for m in record["marks"]) / len(record["marks"])
    return f"{marks_text}\nAverage: {avg:.1f}/100"


def analyze_marks_for_prompt(name: str):
    """Generate a gentle follow-up prompt if marks are low."""
    record = marks_col.find_one({"name": name})
    if not record or not record.get("marks"):
        return ""

    low_subjects = [m["subject"] for m in record["marks"] if m["marks"] < 50]
    if not low_subjects:
        return ""

    subjects_list = ", ".join(low_subjects)
    return (
        f"It seems {name} has scored relatively low marks in {subjects_list}. "
        "Ask gently if there were any particular difficulties, distractions, or worries affecting study time."
    )

def analyze_emotions(conversation):
    """Return emotion analysis for each user message in the conversation."""
    analyzed = []
    for turn in conversation:
        if "user" in turn:
            text = turn["user"]
            try:
                scores = emotion_classifier(text)
                analyzed.append({"text": text, "emotions": scores})
            except Exception as e:
                analyzed.append({"text": text, "error": str(e)})
    return analyzed


# ─────────────────────────────
#  TOPIC TRACKER
# ─────────────────────────────
class TopicTracker:
    def __init__(self):
        self.last_area = None
        self.repeat_count = 0
    def detect_area(self, text):
        """
        Detects the dominant context area from user text.
        Priority: family > school > peer_relations > emotional_functioning > others
        """
        t = text.lower().strip()

        # --- FAMILY CONTEXT ---
        family_keywords = [
            "father", "mother", "dad", "mom", "parents", "brother", "sister",
            "home", "house", "family", "discipline", "argue", "fight", "beat",
            "shout", "yell", "scold", "care", "love", "support"
        ]
        if any(k in t for k in family_keywords):
            return "family"

        # --- SCHOOL CONTEXT ---
        school_keywords = [
            "teacher", "school", "exam", "homework", "class", "subject",
            "study", "grades", "marks", "result", "report card", "principal",
            "homework", "assignment", "tutor", "lecture", "college"
        ]
        if any(k in t for k in school_keywords):
            return "school"

        # --- PEER RELATIONS CONTEXT ---
        peer_keywords = [
            "friend", "friends", "best friend", "classmates", "group",
            "peer", "play", "bully", "tease", "ignore", "talk to", "alone",
            "rejected", "social", "party", "hangout"
        ]
        if any(k in t for k in peer_keywords):
            return "peer_relations"

        # --- EMOTIONAL FUNCTIONING CONTEXT ---
        emotional_keywords = [
            "sad", "angry", "upset", "scared", "fear", "depressed",
            "happy", "lonely", "cry", "hopeless", "worthless",
            "anxious", "panic", "frustrated", "nervous", "tired"
        ]
        if any(k in t for k in emotional_keywords):
            return "emotional_functioning"

        # --- MEDICAL / PHYSICAL CONTEXT ---
        medical_keywords = [
            "headache", "stomach", "sick", "pain", "injury", "tired",
            "sleep", "fatigue", "ill", "medicine", "doctor", "health"
        ]
        if any(k in t for k in medical_keywords):
            return "medical_physical"

        # --- ENVIRONMENTAL / STRESSOR CONTEXT ---
        environmental_keywords = [
            "money", "financial", "neighbour", "noise", "move", "migration",
            "violence", "accident", "loss", "death", "disaster", "community",
            "problem at home", "problem in area"
        ]
        if any(k in t for k in environmental_keywords):
            return "environmental_stressors"

        # --- COGNITIVE / BEHAVIORAL CONTEXT ---
        cognitive_keywords = [
            "focus", "concentrate", "memory", "think", "thoughts",
            "obsess", "compulsion", "forget", "attention", "mind"
        ]
        if any(k in t for k in cognitive_keywords):
            return "cognitive_behavioral"

        # --- SOCIAL SUPPORT CONTEXT ---
        social_keywords = [
            "counsellor", "teacher support", "relative", "friend help",
            "safe", "guidance", "trust", "mentor", "support", "help me"
        ]
        if any(k in t for k in social_keywords):
            return "social_support"

        # --- DEVELOPMENTAL HISTORY CONTEXT ---
        developmental_keywords = [
            "childhood", "speech", "language", "feeding", "sleep pattern",
            "milestone", "toilet training", "growth", "temper tantrum"
        ]
        if any(k in t for k in developmental_keywords):
            return "developmental_history"

        # --- DEFAULT FALLBACK ---
        return "other"

    def update(self, text):
        area = self.detect_area(text)
        if area == self.last_area:
            self.repeat_count += 1
        else:
            self.repeat_count = 1
            self.last_area = area
        return area, self.repeat_count

tracker = TopicTracker()

# ─────────────────────────────
#  EMBEDDING MODEL
# ─────────────────────────────
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    toks = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**toks)
        return out.last_hidden_state.mean(dim=1).numpy()

# ─────────────────────────────
#  BUILD RAG CONTEXT
#  (load your empathic JSON dataset before running)
# ─────────────────────────────
import json
DATA_PATH = "combined_dataset.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)["dataset"]

CORPUS, META = [], []
for d in DATA:
    text = f"{d['category']} | {d['system_prompt']} {d['empathic_response']} {d['empathic_question']} {d['next_question']}"
    CORPUS.append(text)
    META.append(d)

CORPUS_EMB = embed_texts(CORPUS)

def retrieve_context(query, top_k=3):
    q_emb = embed_texts([query])
    sims = cosine_similarity(q_emb, CORPUS_EMB)[0]
    idxs = sims.argsort()[-top_k:][::-1]
    return [META[i] for i in idxs]

# ─────────────────────────────
#  LANGCHAIN MEMORY INTEGRATION
# ─────────────────────────────
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(["start"], embedder)

summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    input_key="question"  # <── crucial line
)
buffer_memory = ConversationBufferMemory(return_messages=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    retriever=retriever,
    memory=summary_memory
)

# ─────────────────────────────
#  RESPONSE GENERATOR
# ─────────────────────────────
def generate_response(user_text: str, name: str | None, question_index=1, max_questions=10):
    """
    Generate an empathetic AI response using LangChain and adaptive context.
    Integrates academic marks when discussion relates to school or studies.
    """
    if not user_text:
        return "Could you share a bit more about how you're feeling?"

    # --- Context detection ---
    area = tracker.detect_area(user_text)
    all_areas = list(CONTEXT_AREAS.keys())
    next_index = (question_index - 1) % len(all_areas)
    current_area = all_areas[next_index]
    keywords = ", ".join(CONTEXT_AREAS.get(current_area, [])[:5])

    # --- Base tone setup ---
    progress_ratio = question_index / max_questions
    tone_note = (
        f"You are {int(progress_ratio*100)}% through the conversation. "
        "Early stages should sound exploratory, later stages more reflective."
    )

    # --- Academic context injection ---
    if area == "school" or any(k in user_text.lower() for k in ["exam", "study", "marks", "grade"]):
        marks_info = get_user_marks(name or "", marks_col)
        if marks_info:
            marks_context = f"\nHere are {name}'s current marks:\n{marks_info}\n"
        else:
            marks_context = f"\nNo stored marks found for {name}.\n"
    else:
        marks_context = ""

    # --- Construct the adaptive prompt ---
    combined_prompt = f"""
You are an empathetic counselor for adolescents aged 10–17.

Conversation focus: {current_area}
Detected theme: {area}
Keywords: {keywords}
{tone_note}
{marks_context}

Child said: "{user_text}"

Respond with one compassionate reflection and one follow-up question related to the above focus.
If marks context is provided, discuss their performance gently — acknowledge effort, avoid blame.
"""
    try:
        result = rag_chain.invoke({"question": combined_prompt, "chat_history": []})
        response = result.get("answer", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        response = f"(Engine error: {str(e)})"

    if not isinstance(response, str):
        response = str(response)
    return response.strip()



# ─────────────────────────────
#  SAVE CHAT TO MONGODB
# ─────────────────────────────
def save_conversation(conversation, user_name: str | None):
    analyzed_conversation = []
    for turn in conversation:
        if "user" in turn:
            text = turn["user"]
            try:
                scores = emotion_classifier(text)
                if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], list):
                    scores = scores[0]

                if isinstance(scores, list):
                    emotion_dict = {}
                    for item in scores:
                        if isinstance(item, dict):
                            label = str(item.get("label", "unknown"))
                            val = item.get("score", 0.0)
                            try:
                                emotion_dict[label] = round(float(val), 4)
                            except (TypeError, ValueError):
                                emotion_dict[label] = 0.0
                    turn["emotion_scores"] = emotion_dict
                else:
                    turn["emotion_scores"] = {"error": str(scores)}

            except Exception as e:
                turn["emotion_scores"] = {"error": str(e)}
        analyzed_conversation.append(turn)

    record = {
        "name": user_name or "Unknown",
        "conversation": analyzed_conversation,
        "timestamp": datetime.datetime.now()
    }

    chats_col.insert_one(record)

    # Optional local backup for JSON download
    folder = "chat_sessions"
    os.makedirs(folder, exist_ok=True)
    fname = f"{user_name}_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(folder, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=safe_json)

    print(f"✅ Conversation saved with emotion analysis → {path}")
    return record




# ─────────────────────────────
#  MAIN CHAT LOOP
# ─────────────────────────────
def run_chat():
    print("=== Child Adaptive RAG Mental Health Chat (with Memory) ===")
    print("Type 'exit' to stop manually.\n")

    conversation = []
    max_questions = 10 #for testing purpose we reduced from 30 to 10
    question_count = 0

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() == "exit":
            print("\nAI: That’s okay! We can stop anytime. Take care and remember, you matter.\n")
            break

        reply = generate_response(user, name=None)
        question_count += 1
        conversation.append({"user": user, "ai": reply})
        print(f"\nAI: {reply}\n")

        if question_count >= max_questions:
            goodbye_msg = (
                "We’ve talked about a lot today. "
                "Let’s take a pause for now — I really enjoyed hearing from you. "
                "Remember, you’re doing your best, and that matters. "
                "Goodbye for now, and take care of yourself."
            )
            print(f"\nAI: {goodbye_msg}\n")
            conversation.append({"ai": goodbye_msg})
            break

    return conversation


# # ─────────────────────────────
# #  RUN
# # ─────────────────────────────
# if __name__ == "__main__":
#     convo = run_chat()
#     save_conversation(convo)
