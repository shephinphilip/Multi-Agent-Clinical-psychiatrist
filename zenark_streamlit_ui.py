import streamlit as st
import json, os, random, uuid
from datetime import datetime
from typing import Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import threading, asyncio, concurrent.futures
import zenarck_optimized as backend_async

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, maxPoolSize=5)
db = client["zenark_db"]
marks_col = db["student_marks"]
reports_col = db["reports"]


_backend_loop: Optional[asyncio.AbstractEventLoop] = None
_backend_thread: Optional[threading.Thread] = None
_backend_initialized = False

# --- Force backend DB initialization exactly once ---
_backend_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_backend_loop)
_backend_loop.run_until_complete(backend_async.init_db())

def _ensure_backend_loop() -> asyncio.AbstractEventLoop:
    global _backend_loop, _backend_thread

    if (
        _backend_loop is not None
        and _backend_thread is not None
        and _backend_thread.is_alive()
    ):
        return _backend_loop

    _backend_loop = asyncio.new_event_loop()
    loop = _backend_loop

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _backend_thread = threading.Thread(target=_run_loop, daemon=True)
    _backend_thread.start()
    return loop


def _ensure_backend_initialized():
    global _backend_initialized
    if _backend_initialized:
        return

    loop = _ensure_backend_loop()
    fut = asyncio.run_coroutine_threadsafe(backend_async.init_db(), loop)
    fut.result()

    _backend_initialized = True


def run_coro_sync(coro, timeout: Optional[float] = 60.0):
    _ensure_backend_initialized()
    loop = _ensure_backend_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout)


def generate_response(prompt: str, name: str, question_index: int, max_questions: int, session_id: str):
    try:
        coro = backend_async.generate_response(
            user_text=prompt,
            name=name,
            question_index=question_index,
            max_questions=max_questions,
            session_id=session_id,
            marks_col=backend_async.marks_col
        )
        result = run_coro_sync(coro)
        if isinstance(result, dict):
            return str(result.get("response", ""))
        return str(result)
    except Exception as e:
        return f"Error calling backend.generate_response: {e}"


def save_conversation(conversation: list, user_name: Optional[str], session_id: str):
    try:
        coro = backend_async.save_conversation(conversation, user_name, session_id)
        return run_coro_sync(coro)
    except Exception as e:
        return {"error": f"Save failed: {e}"}


def generate_report(name: str):
    try:
        coro = backend_async.generate_report(name)
        return run_coro_sync(coro, timeout=120.0)
    except Exception as e:
        return {"error": f"Report failed: {e}"}


st.set_page_config(page_title="Zenark Empathy Chat", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    .chat-message {padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;}
    .user-message {background-color: #e3f2fd; text-align: right;}
    .ai-message {background-color: #f5f5f5;}
    .save-button {background-color: #4CAF50; color: white;}
    .marks-input {max-width: 300px;}
    </style>
""", unsafe_allow_html=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "name" not in st.session_state:
    st.session_state.name = None
if "question_index" not in st.session_state:
    st.session_state.question_index = 1
if "max_questions" not in st.session_state:
    st.session_state.max_questions = 10
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

marks_tab, chat_tab, report_tab = st.tabs(["📝 Mark Entry", "💬 Chat", "📊 Report"])

with marks_tab:
    st.header("📝 Enter Academic Marks")
    st.markdown("Generate and save random marks for six subjects (50-90 range).")

    subjects = ['Maths', 'Physics', 'Chemistry', 'English', 'Hindi', 'Kannada']

    student_name = st.text_input("Student Name", placeholder="Enter student name")

    if st.button("🎲 Generate Random Marks"):
        if student_name:
            marks_list = [{"subject": s, "marks": random.randint(50, 90)} for s in subjects]
            record = {"name": student_name, "marks": marks_list, "timestamp": datetime.now()}
            marks_col.insert_one(record)
            st.success(f"Marks saved for {student_name}")
            st.json(record)
        else:
            st.warning("Please enter a name")

    if st.button("🔍 View Latest Marks"):
        latest = marks_col.find_one(sort=[("_id", -1)])
        if latest:
            st.json(latest)
        else:
            st.info("No marks found.")

with chat_tab:
    st.header("💬 Zenark Empathy Chat")

    if st.session_state.name is None:
        st.info("Please tell me your name to begin.")
        name_input = st.text_input("Your name", placeholder="Enter name")
        if st.button("Start Chat") and name_input.strip():
            st.session_state.name = name_input.strip()
            greeting = f"Hi {st.session_state.name}, how is your day today?"
            st.session_state.conversation.append({"ai": greeting})
            st.session_state.question_index = 1
            st.rerun()
    else:
        for msg in st.session_state.conversation:
            with st.chat_message("user" if "user" in msg else "assistant"):
                display_msg = msg.get("user") or msg.get("ai", "")
                if "user" not in msg and "\n\nSource: " in display_msg:
                    display_msg = display_msg.split("\n\nSource: ")[0]
                st.write(display_msg)

        if prompt := st.chat_input("What's on your mind today?"):
            st.session_state.conversation.append({"user": prompt})
            st.session_state.question_index += 1

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking empathetically..."):
                    reply = generate_response(
                        prompt,
                        st.session_state.name,
                        st.session_state.question_index,
                        st.session_state.max_questions,
                        st.session_state.session_id
                    )
                    if "\n\nSource: " in reply:
                        clean_reply = reply.split("\n\nSource: ")[0].strip()
                    else:
                        clean_reply = reply.strip()
                    st.write(clean_reply)
                    st.session_state.conversation.append({"ai": clean_reply})

            st.session_state.question_index += 1
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Conversation"):
                record = save_conversation(
                    st.session_state.conversation,
                    st.session_state.name,
                    st.session_state.session_id
                )
                if isinstance(record, dict) and record.get("error"):
                    st.error(record.get("error"))
                else:
                    st.success("Conversation saved.")
                    st.json(record)
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation = []
                st.session_state.question_index = 1
                st.rerun()

with report_tab:
    st.header("📊 Generate Empathy Report")
    name_input = st.text_input("Enter name", value=st.session_state.name or "")

    if st.button("Generate Report"):
        if not name_input:
            st.warning("Enter a name.")
        else:
            with st.spinner("Generating reflective report..."):
                report_data = generate_report(name_input.strip())
                if isinstance(report_data, dict) and report_data.get("error"):
                    st.error(report_data.get("error"))
                else:
                    st.success("Report generated successfully.")
                    st.json(report_data)

st.markdown("---")
st.caption("Powered by Zenark Empathy AI")
