"""
Zenark - AI Companion (Modern UI Edition)
──────────────────────────────────────────────
A clean, emotional, and production-ready chatbot interface
connected to ZenarkBrain.
"""

import streamlit as st
import asyncio
import logging
import os
import uuid
from typing import Dict, Any, List
from datetime import datetime

# ---------------------------------------
# Logging
# ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("zenark_ui")

# ---------------------------------------
# Local Imports
# ---------------------------------------
from zenark_brain import ZenarkBrain
from services.zenark_db_cloud import (
    create_user,
    get_user,
    save_conversation,
    get_conversation_history,
)

import nest_asyncio
nest_asyncio.apply()

# ---------------------------------------
# Streamlit Config
# ---------------------------------------
st.set_page_config(
    page_title="Zenark AI — Emotional Clarity",
    page_icon="🧘",
    layout="centered",
)

# ---------------------------------------
# State Initialization
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
# ✨ Modern UI Styling (Glass + Gradient)
# ---------------------------------------
st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at 20% 20%, #0f172a 0%, #020617 100%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.zenark-title {
    text-align: center;
    font-size: 2.3rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}
.zenark-subtitle {
    text-align: center;
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.chat-message {
    padding: 1rem 1.3rem;
    border-radius: 18px;
    margin: 0.6rem 0;
    font-size: 1rem;
    line-height: 1.55;
    width: fit-content;
    max-width: 80%;
    word-wrap: break-word;
    animation: fadeIn 0.3s ease-in;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.user-message {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    margin-left: auto;
    border: 1px solid rgba(255,255,255,0.1);
}
.zen-message {
    backdrop-filter: blur(12px);
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    color: #f8fafc;
    margin-right: auto;
}
.stTextInput>div>div>input {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
}
.stTextInput>div>div>input:focus {
    border: 1px solid #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96,165,250,0.4);
}
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.04);
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
}
.welcome-box {
    background: rgba(30,41,59,0.55);
    padding: 2rem;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    margin: 2rem 0;
}
.welcome-box h2 {
    color: #ffffff;
    margin-bottom: 0.5rem;
}
.welcome-box p {
    color: #cbd5e1;
    font-size: 1.05rem;
}
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Render Chat
# ---------------------------------------
def render_chat():
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "zen-message"
        st.markdown(f"<div class='chat-message {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# ---------------------------------------
# Load Conversation
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
# Start Conversation
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
# Process User Message (with hallucination guard)
# ---------------------------------------
async def process_message(user_input: str):
    username = st.session_state.username
    session_id = st.session_state.session_id
    brain = st.session_state.brain

    if not user_input or not user_input.strip():
        st.warning("⚠️ Please type something meaningful.")
        return

    cleaned_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": cleaned_input})

    user_context = await get_user(username)
    history = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]

    try:
        response = await brain.continue_conversation(
            username=username,
            session_id=session_id,
            user_message=cleaned_input,
            user_context=user_context or {},
            conversation_history=history
        )

        output = response.get("output", {}) if isinstance(response, dict) else {}
        reply = (
            output.get("final_prompt")
            or response.get("fallback_response")
            or "I'm here with you."
        )

        # 🧠 Filter hallucinated or duplicate lines
        if st.session_state.messages and reply.strip():
            last_message = st.session_state.messages[-1]["content"]
            if reply.strip() == last_message.strip():
                reply = "It sounds painful when someone close treats you harshly. How does that usually make you feel inside?"

        if any(word in reply.lower() for word in ["assistant", "system prompt", "llm output"]):
            reply = "I'm here with you — let's focus on what you're feeling right now."

    except Exception as e:
        logger.exception("Error in continue_conversation")
        reply = "Something went wrong. Let’s try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    await save_conversation(username, session_id, "User", cleaned_input)
    await save_conversation(username, session_id, "Zen", reply)

# ---------------------------------------
# UI Layout
# ---------------------------------------
st.markdown("<h1 class='zenark-title'>Zenark AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='zenark-subtitle'>Your reflective companion for emotional clarity</p>", unsafe_allow_html=True)

# Sidebar
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

# Main chat
if not st.session_state.chat_started:
    st.markdown('<div class="welcome-box">', unsafe_allow_html=True)
    st.markdown("<h2>💬 Welcome to Zenark</h2>", unsafe_allow_html=True)
    st.markdown("<p>A calm, safe space for emotional reflection and support.</p>", unsafe_allow_html=True)
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
