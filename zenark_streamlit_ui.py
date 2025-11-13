# ============================================================
# Zenark Empathy AI - Streamlit Frontend UI
# ============================================================
# Interactive web interface for empathetic AI conversations.
# Bridges Streamlit's synchronous world with async FastAPI backend.
#
# KEY ARCHITECTURE:
# - Streamlit: Synchronous framework, reruns entire script on every interaction
# - Backend: Fully async (Motor for MongoDB, LangGraph for orchestration)
# - Bridge: Persistent event loop in background thread for Motor lifecycle
#
# CRITICAL ISSUE & SOLUTION:
# Problem: Motor (async MongoDB) requires a PERSISTENT event loop
# - asyncio.run() creates a loop, runs code, then CLOSES the loop
# - Streamlit reruns constantly (every button click, input change, etc.)
# - Motor tries to reuse a closed loop -> RuntimeError: Event loop is closed
# Solution: One event loop running FOREVER in daemon thread
# - Loop never closes (daemon=True ensures it dies when app closes)
# - Motor connections stay alive across all Streamlit reruns
# - All async backend calls reuse same loop via asyncio.run_coroutine_threadsafe()
# ============================================================

import streamlit as st
import json, os, random, uuid
from datetime import datetime
from typing import Optional
from pymongo import MongoClient  # Sync Mongo client (for marks tab - simple ops)
from dotenv import load_dotenv
import threading, asyncio, concurrent.futures  # For persistent event loop
import zenarck_optimized as backend_async  # Import async backend module

# ============================================================
# ENVIRONMENT & CONNECTION SETUP
# ============================================================
load_dotenv()  # Load environment variables from .env file
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")  # Default fallback

# Sync MongoDB client for marks tab (separate from backend's async Motor client)
# Both connect to same DB but via different drivers
# Sync driver is fine here: marks reads/writes are simple, not performance-critical
client = MongoClient(MONGO_URI, maxPoolSize=5)  # Sync Mongo, small pool (Streamlit bound)
db = client["zenark_db"]  # Main database
marks_col = db["student_marks"]  # Collection: student marks (used by marks tab)
reports_col = db["reports"]  # Collection: generated reports (not used in UI, backend only)

# ============================================================
# PERSISTENT EVENT LOOP FOR ASYNC BACKEND
# ============================================================
# This is the KEY to making async Motor work reliably with Streamlit
# ============================================================

_backend_loop: Optional[asyncio.AbstractEventLoop] = None  # Event loop object
_backend_thread: Optional[threading.Thread] = None  # Thread running loop
_backend_initialized = False  # Track if init_db() was called


def _ensure_backend_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create the persistent event loop running in a background thread.
    
    IMPORTANT: This loop runs FOREVER. It never closes.
    This keeps Motor (async MongoDB) connections alive across Streamlit reruns.
    
    Returns:
        asyncio.AbstractEventLoop: The persistent loop (same object each time)
    
    Flow:
    1. First call: create new loop, start daemon thread, return loop
    2. Subsequent calls: return existing loop (thread still alive)
    3. If thread died: create new loop + thread (failsafe, shouldn't happen)
    """
    global _backend_loop, _backend_thread

    # Reuse existing loop if thread is still running
    if (
        _backend_loop is not None
        and _backend_thread is not None
        and _backend_thread.is_alive()
    ):
        return _backend_loop

    # Create new loop for background thread
    _backend_loop = asyncio.new_event_loop()
    loop = _backend_loop  # Local ref (helps with type narrowing)

    def _run_loop():
        """Infinite loop runner - runs forever in background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()  # Blocks forever (until thread dies as daemon)

    # Start daemon thread (dies when app closes)
    _backend_thread = threading.Thread(target=_run_loop, daemon=True)
    _backend_thread.start()
    
    return loop


def _ensure_backend_initialized():
    """
    Ensure backend MongoDB collections are initialized.
    Calls backend_async.init_db() if not already done.
    
    Why separate from _ensure_backend_loop()?
    - Loop lifecycle and DB initialization are independent
    - DB only needs init once; loop might restart
    - Allows separate error handling
    """
    global _backend_initialized
    
    if _backend_initialized:
        return  # Already initialized, skip
    
    loop = _ensure_backend_loop()  # Get persistent loop
    # Schedule init_db coroutine on background loop
    fut = asyncio.run_coroutine_threadsafe(backend_async.init_db(), loop)
    fut.result()  # Block until init completes
    
    _backend_initialized = True  # Mark done


def run_coro_sync(coro, timeout: Optional[float] = 60.0):
    """
    Bridge between Streamlit (sync) and backend (async).
    Runs a coroutine on the persistent event loop and returns result.
    
    Args:
        coro: Coroutine object to execute
        timeout: Max seconds to wait (default 60s)
    
    Returns:
        Coroutine's return value
    
    Raises:
        concurrent.futures.TimeoutError: If exceeds timeout
        Any exception raised by coroutine is re-raised
    
    How:
    1. Ensure backend DB is initialized
    2. Get persistent event loop
    3. Schedule coroutine on that loop
    4. Wait for result with timeout
    5. Return result
    """
    _ensure_backend_initialized()  # Check DB is initialized
    loop = _ensure_backend_loop()  # Get the persistent loop
    # Submit coroutine to run on background loop
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout)  # Wait and get result


# ============================================================
# SYNCHRONOUS WRAPPERS AROUND ASYNC BACKEND FUNCTIONS
# ============================================================
# These functions hide async/await complexity from Streamlit UI
# ============================================================


def generate_response(
    prompt: str, 
    name: str, 
    question_index: int, 
    max_questions: int, 
    session_id: str
) -> str:
    """
    Generate empathetic AI response to user input.
    This is the main chat function: user message -> AI response.
    
    Args:
        prompt: User's message/input text
        name: User's name (for personalization)
        question_index: Current question number in session
        max_questions: Max questions before suggesting topic change
        session_id: Unique conversation session ID (for MongoDB persistence)
    
    Returns:
        str: AI's response text, ready for UI display.
             On error, returns error message string.
    
    Backend Pipeline:
    1. Sentiment classification (positive/negative/neutral)
    2. Category detection (school_stress, family, etc.)
    3. RAG retrieval (find similar conversations)
    4. LLM response generation (with LangGraph orchestration)
    5. Return response dict
    
    This wrapper extracts just the response text for display.
    """
    try:
        # Build coroutine (args match backend signature)
        coro = backend_async.generate_response(
            user_text=prompt,
            name=name,
            question_index=question_index,
            max_questions=max_questions,
            session_id=session_id,
            marks_col=backend_async.marks_col,  # Pass backend's async Mongo collection
        )
        
        # Run coroutine via persistent loop
        result = run_coro_sync(coro)
        
        # Extract response text from result dict
        if isinstance(result, dict):
            return str(result.get("response", ""))
        return str(result)
        
    except Exception as e:
        # Return user-friendly error (doesn't crash Streamlit)
        return f"Error generating response: {e}"


def save_conversation(
    conversation: list, 
    user_name: Optional[str], 
    session_id: str
) -> dict:
    """
    Save complete conversation to MongoDB.
    Stores all user messages, AI responses, metadata, and creates backup JSON.
    
    Args:
        conversation: List of message dicts [{\"user\": str}, {\"ai\": str}, ...]
        user_name: User's name (None is valid, stored as \"Unknown\")
        session_id: Unique session ID for this conversation
    
    Returns:
        dict: Saved record with MongoDB _id on success, or {\"error\": str} on failure
    
    Persistence:
    - Stores to MongoDB 'chat_sessions' collection
    - Also writes backup JSON to local 'chat_sessions/' folder
    - Records timestamp for session tracking
    - Used later for report generation
    
    Error Handling:
    - Returns error dict (not exception) for graceful UI error display
    """
    try:
        # Build coroutine
        coro = backend_async.save_conversation(conversation, user_name, session_id)
        # Run and return result
        return run_coro_sync(coro)
    except Exception as e:
        return {"error": f"Save failed: {e}"}


def generate_report(name: str) -> dict:
    """
    Generate multi-agent AI report from saved conversation.
    Uses AutoGen framework with specialized agent roles.
    
    Args:
        name: User's name (to look up conversation)
    
    Returns:
        dict: Generated report with agent outputs, or {\"error\": str} on failure
    
    Report Components (on success):
    - \"InterventionSpecialistAgent\": Personalized wellness guide
    - \"DataAnalystAgent\": Strengths & weaknesses analysis (JSON parsed)
    - \"RoutinePlannerAgent\": 7-day self-care plan
    
    Timeout: 120 seconds (report generation is expensive)
    - Multi-agent orchestration
    - LLM calls for each agent
    - JSON parsing and aggregation
    
    Error Cases:
    - No conversation found for user -> error
    - MongoDB not initialized -> error
    - LLM API failures -> error
    """
    try:
        # Build coroutine
        coro = backend_async.generate_report(name)
        # Run with longer timeout (report generation is slow)
        return run_coro_sync(coro, timeout=120.0)
    except Exception as e:
        return {"error": f"Report failed: {e}"}


# ============================================================
# STREAMLIT UI CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zenark Empathy Chat",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat messages
st.markdown("""
    <style>
    .chat-message {padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;}
    .user-message {background-color: #e3f2fd; text-align: right;}
    .ai-message {background-color: #f5f5f5;}
    .save-button {background-color: #4CAF50; color: white;}
    .marks-input {max-width: 300px;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# STREAMLIT SESSION STATE
# ============================================================
# Session state persists across reruns (like browser local storage)
# Streamlit reruns entire script on every interaction (button, input, etc.)
# ============================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of messages: [{\"user\": str}, {\"ai\": str}, ...]
if "name" not in st.session_state:
    st.session_state.name = None  # User's name (None before \"Start Chat\")
if "question_index" not in st.session_state:
    st.session_state.question_index = 1  # Current Q number (for progress tracking)
if "max_questions" not in st.session_state:
    st.session_state.max_questions = 10  # Max Q before suggesting topic change
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Unique session ID

# ============================================================
# UI TABS
# ============================================================
# Three main sections: marks entry (admin), chat (user), report (analysis)
marks_tab, chat_tab, report_tab = st.tabs(["📝 Mark Entry", "💬 Chat", "📊 Report"])

# --- TAB 1: MARK ENTRY ---
# Admin interface to manage student marks (used by backend's fetch_marks_node)
with marks_tab:
    st.header("📝 Enter Academic Marks")
    st.markdown("Generate and save random marks for six subjects (50-90 range).")
    
    subjects = ['Maths', 'Physics', 'Chemistry', 'English', 'Hindi', 'Kannada']
    
    student_name = st.text_input("Student Name", placeholder="Enter student name")

    if st.button("🎲 Generate Random Marks"):
        if student_name:
            # Generate random marks for each subject
            marks_list = [{"subject": s, "marks": random.randint(50, 90)} for s in subjects]
            record = {"name": student_name, "marks": marks_list, "timestamp": datetime.now()}
            marks_col.insert_one(record)  # Store in MongoDB
            st.success(f"Marks saved for {student_name}")
            st.json(record)
        else:
            st.warning("Please enter a name")

    if st.button("🔍 View Latest Marks"):
        # Retrieve and display most recent marks
        latest = marks_col.find_one(sort=[("_id", -1)])
        if latest:
            st.json(latest)
        else:
            st.info("No marks found.")

# --- TAB 2: CHAT ---
# Main user interface for multi-turn empathetic conversation
with chat_tab:
    st.header("💬 Zenark Empathy Chat")

    # Stage 1: Get user's name (first interaction)
    if st.session_state.name is None:
        st.info("Please tell me your name to begin.")
        name_input = st.text_input("Your name", placeholder="Enter name")
        if st.button("Start Chat") and name_input.strip():
            # Initialize session
            st.session_state.name = name_input.strip()
            greeting = f"Hi {st.session_state.name}, how is your day today?"
            st.session_state.conversation.append({"ai": greeting})
            st.session_state.question_index = 1
            st.rerun()
    
    # Stage 2: Chat (after name entered)
    else:
        # Display all previous messages
        for msg in st.session_state.conversation:
            with st.chat_message("user" if "user" in msg else "assistant"):
                display_msg = msg.get("user") or msg.get("ai", "")
                # Remove RAG source citations from display (kept in backend for logging)
                if "user" not in msg and "\n\nSource: " in display_msg:
                    display_msg = display_msg.split("\n\nSource: ")[0]
                st.write(display_msg)

        # Chat input box - awaits user message
        if prompt := st.chat_input("What's on your mind today?"):
            # Record user message
            st.session_state.conversation.append({"user": prompt})
            st.session_state.question_index += 1

            # Display user message in UI
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking empathetically..."):
                    # Call backend to generate response
                    # Backend runs sentiment -> category -> RAG -> LLM
                    reply = generate_response(
                        prompt,
                        st.session_state.name,
                        st.session_state.question_index,
                        st.session_state.max_questions,
                        st.session_state.session_id
                    )
                    
                    # Clean up response for display (remove RAG citations)
                    if "\n\nSource: " in reply:
                        clean_reply = reply.split("\n\nSource: ")[0].strip()
                    else:
                        clean_reply = reply.strip()
                    
                    # Show AI response to user
                    st.write(clean_reply)
                    # Store in history for later save/report
                    st.session_state.conversation.append({"ai": clean_reply})

            st.session_state.question_index += 1
            st.rerun()

        # Action buttons below conversation
        col1, col2 = st.columns(2)
        
        # Save conversation to MongoDB + backup file
        with col1:
            if st.button("💾 Save Conversation"):
                record = save_conversation(
                    st.session_state.conversation,
                    st.session_state.name,
                    st.session_state.session_id
                )
                # Display result to user
                if isinstance(record, dict) and record.get("error"):
                    st.error(record.get("error"))
                else:
                    st.success("Conversation saved.")
                    st.json(record)  # Show saved record with MongoDB _id
        
        # Clear current conversation and start fresh
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation = []
                st.session_state.question_index = 1
                st.rerun()

# --- TAB 3: REPORT GENERATION ---
# Generate multi-agent AI report from saved conversation
with report_tab:
    st.header("📊 Generate Empathy Report")
    
    # Input field for user name (pre-fill if already chatting)
    name_input = st.text_input("Enter name", value=st.session_state.name or "")

    if st.button("Generate Report"):
        if not name_input:
            st.warning("Enter a name.")
        else:
            # Generate report (can take 30-120s)
            # Uses AutoGen multi-agent framework
            with st.spinner("Generating reflective report..."):
                report_data = generate_report(name_input.strip())
                # Display result
                if isinstance(report_data, dict) and report_data.get("error"):
                    st.error(report_data.get("error"))
                else:
                    st.success("Report generated successfully.")
                    st.json(report_data)  # Display full report

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Powered by Zenark Empathy AI")
