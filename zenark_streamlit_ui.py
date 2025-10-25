"""
Zenark - Clean AI Companion Chat UI
──────────────────────────────────────────────
Minimal, calm, and production-ready chatbot UI.
"""

import streamlit as st
import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List

# ---------------------------------------
# Logging & Environment Setup
# ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("zenark_ui")

# Local imports
from zenark_brain import ZenarkBrain
from services.zenark_db_cloud import (
    create_user,
    get_user,
    save_conversation,
    get_conversation_history,
)

# Allow nested async calls in Streamlit
import nest_asyncio
nest_asyncio.apply()

# ---------------------------------------
# Streamlit Page Config
# ---------------------------------------
st.set_page_config(
    page_title="Zenark - AI Companion",
    page_icon="🧘",
    layout="centered",
)

# ---------------------------------------
# Initialize Session State
# ---------------------------------------
if "brain" not in st.session_state:
    st.session_state.brain = ZenarkBrain()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "username" not in st.session_state:
    st.session_state.username = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# ---------------------------------------
# Styling - Minimal Zen UI
# ---------------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0E1117 !important;
    color: #E4E6EB !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.zenark-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}

.zenark-subtitle {
    text-align: center;
    font-size: 1rem;
    color: #94A3B8;
    margin-bottom: 1.5rem;
}

.chat-message {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-size: 1rem;
    line-height: 1.5;
    width: fit-content;
    max-width: 80%;
    word-wrap: break-word;
    animation: fadeIn 0.25s ease-in;
}
.user-message {
    background-color: #2563eb;
    color: #ffffff;
    margin-left: auto;
}
.zen-message {
    background-color: #374151;
    color: #f1f5f9;
    margin-right: auto;
}
.stTextInput>div>div>input {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.7rem !important;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #1E3A8A;
    transform: scale(1.02);
}
.welcome-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin: 2rem 0;
    border: 1px solid #334155;
}
.welcome-box h2 {
    color: #ffffff;
    margin-bottom: 0.5rem;
}
.welcome-box p {
    color: #cbd5e1;
    font-size: 1.1rem;
}
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Chat Renderer
# ---------------------------------------
def render_chat():
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "zen-message"
        st.markdown(f"<div class='chat-message {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# ---------------------------------------
# Load Conversation History
# ---------------------------------------
async def load_history(username: str, session_id: str):
    history = await get_conversation_history(username, session_id, limit=100)
    for msg in history:
        role = "user" if msg["speaker"] == "User" else "assistant"
        st.session_state.messages.append({
            "role": role,
            "content": msg["message"]
        })

# ---------------------------------------
# Start Conversation (Greeting)
# ---------------------------------------
async def start_conversation():
    username = st.session_state.username.strip()
    if not username:
        st.warning("Please enter your name.")
        return False

    brain = st.session_state.brain
    session_id = st.session_state.session_id

    await create_user(username)
    await load_history(username, session_id)

    try:
        result = await brain.start_conversation(
            username=username,
            session_id=session_id,
            user_context={"username": username},
            conversation_history=st.session_state.messages,
        )
        greeting = result.get("greeting", "Hello! How are you feeling today?")
    except Exception as e:
        logger.exception("Failed to get greeting")
        greeting = "Hi there — I'm here to listen."

    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.chat_started = True
    return True

# ---------------------------------------
# Process User Message
# ---------------------------------------
async def process_message(user_input: str):
    username = st.session_state.username
    session_id = st.session_state.session_id
    brain = st.session_state.brain

    st.session_state.messages.append({"role": "user", "content": user_input})

    user_context = await get_user(username)
    history = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]

    try:
        response = await brain.continue_conversation(
            username=username,
            session_id=session_id,
            user_message=user_input,
            user_context=user_context or {},
            conversation_history=history
        )

        # 🌿 Detect Wellness Phase
        if response.get("output", {}).get("phase") == "wellness":
            wellness_message = response["output"].get("message", "")
            st.session_state.messages.append({"role": "assistant", "content": wellness_message})
            st.session_state["phase"] = "wellness"
            st.session_state["wellness_message"] = wellness_message
            return

        reply = (
            response.get("output", {}).get("final_prompt")
            or response.get("fallback_response", "I'm here with you.")
        )

    except Exception as e:
        logger.exception("Error in continue_conversation")
        reply = "Something went wrong. Let’s try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Save both messages
    await save_conversation(username, session_id, "User", user_input)
    await save_conversation(username, session_id, "Zen", reply)


# ---------------------------------------
# UI Layout
# ---------------------------------------
st.markdown("<h1 class='zenark-title'>Zenark AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='zenark-subtitle'>Your reflective companion for emotional clarity</p>", unsafe_allow_html=True)

# Sidebar tools
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if st.button("🔄 New Conversation"):
        st.session_state.clear()
        st.rerun()
    if st.button("🗑️ Delete My Data"):
        if st.session_state.get("username"):
            from services.zenark_db_cloud import delete_user
            asyncio.run(delete_user(st.session_state.username))
            st.success("✅ Your data has been deleted.")
            st.session_state.clear()
            st.rerun()
    st.markdown("---")
    st.markdown("""
    #### 🆘 Crisis Resources  
    - 988: Suicide & Crisis Lifeline  
    - 741741: Crisis Text Line (text **HOME**)  
    - 911: Emergency Services  
    """)

# Main chat area
if not st.session_state.chat_started:
    st.markdown('<div class="welcome-box">', unsafe_allow_html=True)
    st.markdown("<h2>💬 Welcome to Zenark</h2>", unsafe_allow_html=True)
    st.markdown("<p>A safe space for emotional exploration and clarity.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    name_input = st.text_input("What should I call you?", value=st.session_state.username)
    if st.button("Start Chat 🚀"):
        if name_input.strip():
            st.session_state.username = name_input.strip()
            asyncio.run(start_conversation())
            st.rerun()
        else:
            st.warning("Please enter your name.")
else:
    render_chat()
    user_input = st.chat_input("Type your message here...")
    if user_input:
        asyncio.run(process_message(user_input))
        st.rerun()
