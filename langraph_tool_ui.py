import streamlit as st
import requests
import json
from datetime import datetime
import uuid
import base64



# Page config
st.set_page_config(page_title="Zenark Mental Health Bot", page_icon="🧠", layout="wide")

# Title and description
st.title("Zenark - Your Mental Health Companion")
st.markdown(
    """
    A supportive chatbot for students dealing with stress, exams, emotions, and more.  
    Share what's on your mind—I'm here to listen without judgment. 💙
    """
)

# Sidebar for user details and controls
with st.sidebar:
    st.header("Session Settings")
    
    # User name input
    user_name = st.text_input("Your Name", placeholder="Enter your name (optional)")
    
    # Token input (JWT for auth/student_id - use st.secrets in production)
    token = st.text_input(
        "JWT Token", 
        type="password", 
        placeholder="Paste your JWT token here (required for API auth)",
        help="This token should contain 'sub' claim as your student_id."
    )
    
    # Session ID (auto-generated if empty)
    session_id = st.text_input(
        "Session ID", 
        value=str(uuid.uuid4())[:8],  # Short UUID for demo
        help="Unique ID for this chat session. Auto-generated if empty."
    )
    
    # Controls
    if st.button("New Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.markdown("**API Base URL:**")
    # In the sidebar or top
    api_base = st.text_input("Backend URL", value="https://zentool-lvaodv2eb-shephinphilips-projects.vercel.app")
    # Or hardcode: api_base = "https://zentool-lvaodv2eb-shephinphilips-projects.vercel.app"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = session_id or str(uuid.uuid4())

# Update session_id if changed in sidebar
if session_id != st.session_state.session_id:
    st.session_state.session_id = session_id
    st.session_state.messages = []  # Clear messages on new session

current_session_id = st.session_state.session_id

# Validation
if not token:
    st.warning("Please enter a JWT token in the sidebar to start chatting.")
    st.stop()

# Chat interface
st.markdown("### Chat with Zenark")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind today?"):
    # Append user message to history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare API request
    chat_payload = {
        "message": prompt,
        "session_id": current_session_id,
        "token": token
    }
    
    # Call /chat endpoint
    with st.spinner("Zenark is thinking..."):
        try:
            response = requests.post(f"{api_base}/chat", json=chat_payload, timeout=30)
            if response.status_code == 200:
                ai_response = response.json()["response"]
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                error_msg = response.json().get("detail", "Unknown error")
                st.error(f"❌ API Error: {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an issue: {error_msg}"})
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I'm having trouble connecting right now."})
    
    st.rerun()

# Main controls (below chat)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save Conversation", use_container_width=True):
        if not st.session_state.messages:
            st.warning("No messages to save yet.")
        else:
            # Prepare paired conversation format: [{"user": "...", "ai": "..."}, ...]
            paired_conversation = []
            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    user_msg = st.session_state.messages[i]["content"]
                    ai_msg = st.session_state.messages[i + 1]["content"]
                    paired_conversation.append({"user": user_msg, "ai": ai_msg})
                else:
                    # Odd number: unpaired user message
                    paired_conversation.append({"user": st.session_state.messages[i]["content"], "ai": ""})
            
            save_payload = {
                "conversation": paired_conversation,
                "name": user_name or "Anonymous",
                "session_id": current_session_id,
                "token": token
            }
            
            with st.spinner("Saving..."):
                try:
                    save_response = requests.post(f"{api_base}/save_chat", json=save_payload, timeout=30)
                    if save_response.status_code == 200:
                        saved_record = save_response.json()
                        st.success(f"✅ Conversation saved! ID: {saved_record.get('_id', 'N/A')}")
                    else:
                        error_msg = save_response.json().get("detail", "Unknown error")
                        st.error(f"❌ Save Error: {error_msg}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network Error: {str(e)}")

with col2:
    if st.button("Get Distress Score", use_container_width=True):
        score_payload = {"session_id": current_session_id}
        with st.spinner("Analyzing..."):
            try:
                score_response = requests.post(f"{api_base}/score_conversation", json=score_payload, timeout=30)
                if score_response.status_code == 200:
                    score_data = score_response.json()
                    score = score_data["global_distress_score"]
                    st.metric(label="Global Distress Score", value=score, delta=None, help="1 = Minimal distress | 10 = High distress")
                    if "warning" in score_data:
                        st.warning(score_data["warning"])
                else:
                    error_msg = score_response.json().get("detail", "Unknown error")
                    st.error(f"❌ Score Error: {error_msg}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network Error: {str(e)}")

with col3:
    if st.button("View Router Memory", use_container_width=True):
        # Extract student_id from token
        try:
            # Simple base64 decode for 'sub' (assuming no padding issues)
            parts = token.split('.')
            if len(parts) == 3:
                payload = parts[1]
                padded = payload + '=' * (4 - len(payload) % 4)
                decoded = json.loads(base64.urlsafe_b64decode(padded).decode())
                student_id = decoded.get('sub', 'unknown')
                memory_url = f"{api_base}/router-memory/{current_session_id}/{student_id}"
                memory_response = requests.get(memory_url, timeout=30)
                if memory_response.status_code == 200:
                    memory_data = memory_response.json()
                    with st.expander("Router Memory Details"):
                        st.json(memory_data)
                else:
                    st.error(f"❌ Memory Error: {memory_response.status_code}")
            else:
                st.error("Invalid JWT token format.")
        except Exception as e:
            st.error(f"❌ Token Decode Error: {str(e)}")

# Health check info
st.markdown("---")
with st.expander("Backend Status"):
    try:
        health_response = requests.get(f"{api_base}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ Backend is healthy!")
            st.json(health_data)
        else:
            st.error("❌ Backend unhealthy.")
    except:
        st.error("❌ Cannot reach backend.")

# Footer
st.markdown("---")
st.markdown(
    "*Zen, your mental health companion. For crises, contact: +91 9152987821 (24/7). You're not alone.*"
)