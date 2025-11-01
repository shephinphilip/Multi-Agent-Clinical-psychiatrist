#!pip install langchain langchain-core langchain-community==0.3.22 langchain-openai==0.2.0 faiss-cpu

# ============================================================
#  Child Adaptive RAG Chatbot with LangChain Memory Integration
# ============================================================

import re, random, datetime, torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import os
import json, os, datetime
# LangChain imports
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from pymongo import MongoClient
from typing import Optional

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

def get_user_marks(name: str):
    """Retrieve marks for a student by name from MongoDB."""
    record = marks_col.find_one({"name": name})
    if not record:
        return "No marks found for this student."
    marks_text = "\n".join([f"{m['subject']}: {m['marks']}/100" for m in record["marks"]])
    return marks_text

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


# ─────────────────────────────
#  TOPIC TRACKER
# ─────────────────────────────
class TopicTracker:
    def __init__(self):
        self.last_area = None
        self.repeat_count = 0
    def detect_area(self, text):
        t = text.lower()
        for area, kws in CONTEXT_AREAS.items():
            if any(k in t for k in kws):
                return area
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
def generate_response(user_text: str, user_name: Optional[str] = None):
    # store in long-term vector memory
    vectorstore.add_texts([user_text])

    # detect area and repetition
    area, rep = tracker.update(user_text)
    school_reflection = ""
    if area == "school" and user_name:
        school_reflection = analyze_marks_for_prompt(user_name)


    if rep >= 3:
        next_area = random.choice([a for a in CONTEXT_AREAS.keys() if a != area])
        transition_prompt = f"The child has been talking mostly about {area}. Gently move to {next_area} next."
    else:
        transition_prompt = ""

    context_items = retrieve_context(user_text)
    context_text = "\n".join([
        f"Category: {c['category']}\nEmpathy: {c['empathic_response']}\nQuestion: {c['empathic_question']}\nFollow-up: {c['next_question']}"
        for c in context_items
    ])

    # retrieve marks only if name is provided
    marks_info = ""
    if user_name:
        marks_info = get_user_marks(user_name)
        marks_info = f"\nHere are {user_name}'s marks (out of 100):\n{marks_info}\n"

    memory_context = summary_memory.load_memory_variables({}).get("history", "")
    combined_prompt = f"""
You are a compassionate, age-appropriate AI counselor for children aged 10–17.
Use natural, emotionally intelligent language. Avoid repetition.

{marks_info}

Recent conversation summary:
{memory_context}

Relevant empathy dataset context:
{context_text}

Child said: "{user_text}"

{transition_prompt}

{school_reflection}

Respond with one caring reflection and one thoughtful follow-up question.
"""

    result = rag_chain.invoke({"question": combined_prompt, "chat_history": []})
    response = result["answer"] if isinstance(result, dict) else str(result)
    return response.strip()




# ─────────────────────────────
#  SAVE CHAT TO MONGODB
# ─────────────────────────────
def save_conversation(conversation, user_name: str):
    """Save conversation turns into MongoDB with timestamp."""
    try:
        record = {
            "name": user_name,
            "conversation": conversation,
            "timestamp": datetime.datetime.now()
        }
        db["chat_sessions"].insert_one(record)
        print(f"✅ Conversation saved in MongoDB for user: {user_name}")
    except Exception as e:
        print(f"❌ Failed to save conversation: {e}")


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

        reply = generate_response(user)
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
