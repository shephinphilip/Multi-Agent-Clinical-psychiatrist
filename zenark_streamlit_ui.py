from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import datetime, os, json
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from Zenark_Empathy import generate_response, save_conversation

# ============================================================
#  ENV + INIT
# ============================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, maxPoolSize=5)
db = client["zenark_db"]
marks_col = db["student_marks"]
chats_col = db["chat_sessions"]
reports_col = db["reports"]

app = FastAPI(
    title="Zenark Empathy API",
    version="3.0",
    description="FastAPI backend for Zenark Empathy — integrating chatbot, marks, and report generation."
)

# Allow Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific Streamlit URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  SCHEMAS
# ============================================================
class MarksEntry(BaseModel):
    name: str
    marks: List[Dict[str, int]]

class ChatRequest(BaseModel):
    text: str
    name: Optional[str] = "User"
    question_index: int = 1
    max_questions: int = 10

class SaveChatRequest(BaseModel):
    conversation: List[Dict[str, str]]
    name: Optional[str] = None

class ReportRequest(BaseModel):
    name: Optional[str] = None

# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok", "time": datetime.datetime.now().isoformat()}

# ---------------- MARKS ENTRY ----------------
@app.post("/marks")
def submit_marks(entry: MarksEntry):
    record = {
        "name": entry.name,
        "marks": entry.marks,
        "timestamp": datetime.datetime.now()
    }
    marks_col.insert_one(record)
    return {"message": "Marks stored successfully.", "data": record}

# ---------------- EMPATHY CHATBOT ----------------
@app.post("/chat")
def empathy_chat(req: ChatRequest):
    try:
        ai_reply = generate_response(req.text, req.name, req.question_index, req.max_questions)
        return {"response": ai_reply}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- SAVE CHAT ----------------
@app.post("/save_chat")
def save_chat(req: SaveChatRequest):
    try:
        record = save_conversation(req.conversation, req.name)
        return {"message": "Conversation saved successfully.", "data": record}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- AUTOGEN REPORT ----------------
@app.post("/report")
def generate_report(req: ReportRequest):
    try:
        last_chat = chats_col.find_one(sort=[("_id", -1)])
        if not last_chat:
            return JSONResponse(status_code=404, content={"error": "No previous conversation found."})

        conversation_text = "\n".join(
            [f"You: {t.get('user','')}\nAI: {t.get('ai','')}" for t in last_chat["conversation"]]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        summary = llm.invoke([HumanMessage(content=f"Summarize this empathetically:\n{conversation_text}")])
        report = getattr(summary, "content", "(no report generated)")

        doc = {
            "name": last_chat.get("name", "Unknown"),
            "report": report,
            "timestamp": datetime.datetime.now()
        }
        reports_col.insert_one(doc)
        return {"message": "Report generated successfully.", "data": doc}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================================
#  MAIN ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("zenark_api:app", host="0.0.0.0", port=8000, reload=True)
