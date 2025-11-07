import streamlit as st
import requests
import json
import os
import random
import uuid  # For session_id
from datetime import datetime
from typing import List, Dict, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import cast  # For type safety

# Import functions from zenarck_langraph.py
from zenarck_langraph import generate_response, save_conversation

# Load env for Mongo (marks/reports only)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, maxPoolSize=5)
db = client["zenark_db"]
marks_col = db["student_marks"]
reports_col = db["reports"]

# AutoGen report (UI-only)
from autogen_report import generate_autogen_report

# Page config
st.set_page_config(
    page_title="Zenark Empathy Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (unchanged)
st.markdown("""
    <style>
    .chat-message {padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;}
    .user-message {background-color: #e3f2fd; text-align: right;}
    .ai-message {background-color: #f5f5f5;}
    .save-button {background-color: #4CAF50; color: white;}
    .marks-input {max-width: 300px;}
    </style>
""", unsafe_allow_html=True)

# Session state
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
if "show_marks" not in st.session_state:
    st.session_state.show_marks = False
if "show_report" not in st.session_state:
    st.session_state.show_report = False

# Tabs
marks_tab, chat_tab, report_tab = st.tabs(["📝 Mark Entry", "💬 Chat", "📊 Report"])

# Tab 1: Mark Entry (direct Mongo, unchanged)
with marks_tab:
    st.header("📝 Enter Academic Marks")
    st.markdown("Generate and save random marks for six subjects (50-90 range).")
    
    subjects = ['Maths', 'Physics', 'Chemistry', 'English', 'Hindi', 'Kannada']
    
    student_name = st.text_input("Student Name", placeholder="Enter student name")
    
    if st.button("🎲 Generate Random Marks"):
        if student_name:
            marks_list = [{"subject": subject, "marks": random.randint(50, 90)} for subject in subjects]
            record = {"name": student_name, "marks": marks_list, "timestamp": datetime.now()}
            try:
                result = marks_col.insert_one(record)
                st.success(f"Marks saved for {student_name}! ID: {result.inserted_id}")
                st.json(record)
            except Exception as e:
                st.error(f"Failed to save marks: {str(e)}")
        else:
            st.warning("Please enter a student name.")
    
    if st.button("🔍 View Latest Marks"):
        latest = marks_col.find_one(sort=[("_id", -1)])
        if latest:
            st.json(latest)
        else:
            st.info("No marks saved yet.")

# Tab 2: Chat (Direct function calls)
with chat_tab:
    st.header("💬 Zenark Empathy Chat")
    st.markdown("Welcome! Let's start our conversation. 💙")
    
    if st.session_state.name is None:
        st.info("To begin, please tell me your name.")
        name_input = st.text_input("What's your name?", placeholder="Enter your name")
        if st.button("Start Chat") and name_input.strip():
            st.session_state.name = name_input.strip()
            greeting = f"Hi {st.session_state.name}, How is your day today?"
            st.session_state.conversation.append({"ai": greeting})
            st.session_state.question_index = 1
            st.rerun()
    else:
        # Display history
        for msg in st.session_state.conversation:
            with st.chat_message("user" if "user" in msg else "assistant"):
                st.write(msg["user"] if "user" in msg else msg["ai"])
                if "emotion_scores" in msg:
                    scores = msg["emotion_scores"]
                    if isinstance(scores, dict) and "error" not in scores:
                        st.caption(f"Detected emotions: {', '.join([f'{k}: {v:.2f}' for k, v in list(scores.items())[:3]])}")
        
        # Chat input
        if prompt := st.chat_input("What's on your mind today?"):
            st.session_state.conversation.append({"user": prompt})
            st.session_state.question_index += 1
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking empathetically..."):
                        ai_reply = generate_response(
                            prompt,
                            st.session_state.name,
                            st.session_state.question_index,
                            st.session_state.max_questions,
                            st.session_state.session_id
                        )
                        st.write(ai_reply)
                        st.session_state.conversation.append({"ai": ai_reply})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            
            st.rerun()
        
        # Action buttons
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save Conversation", type="primary"):
                try:
                    with st.spinner("Analyzing emotions and saving..."):
                        record = save_conversation(
                            st.session_state.conversation,
                            st.session_state.name,
                            st.session_state.session_id
                        )
                        record_dict = cast(dict, record)
                        st.success(f"Conversation saved! Session ID: {record_dict.get('_id', 'N/A')}")
                        st.json({"summary": f"Saved {len(st.session_state.conversation)} turns for {st.session_state.name}"})
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")
        
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation = []
                st.session_state.question_index = 1
                st.rerun()

# ================================================================
# Tab 3: Report (Integrated with FastAPI /generate_report endpoint)
# ================================================================
with report_tab:
    st.header("📊 Generate Empathy Report")
    st.markdown("Generate a multi-agent reflective report from your saved conversation.")

    name_input = st.text_input("Enter Name to Load Conversation", value=st.session_state.name or "")

    if st.button("🔍 Load Conversation"):
        if name_input:
            st.session_state.name = name_input.strip()
            latest_conv = db["chat_sessions"].find_one(
                {"name": st.session_state.name}, sort=[("_id", -1)]
            )
            if latest_conv and "conversation" in latest_conv:
                st.session_state.conversation = latest_conv["conversation"]
                st.success(f"Loaded latest conversation for {st.session_state.name}")
            else:
                st.warning(f"No chat found for {st.session_state.name}")
        else:
            st.warning("Please enter a name.")

    if not st.session_state.conversation:
        st.warning("No conversation history found. Please chat first.")
    else:
        if st.button("Generate Now"):
            try:
                with st.spinner("Contacting backend to generate multi-agent reflective report..."):
                    # Call FastAPI backend
                    api_url = "http://localhost:8000/generate_report"
                    payload = {"name": st.session_state.name}
                    response = requests.post(api_url, json=payload)

                    if response.status_code != 200:
                        st.error(f"API error: {response.text}")
                    else:
                        report_data = response.json()
                        st.markdown("### AutoGen Reflective Report")

                        # Parse report structure
                        messages = report_data.get("report", [])
                        agent_outputs = {}
                        for msg in messages:
                            role = msg.get("name", "Unknown")
                            content = msg.get("content", "").strip()
                            if content and role in agent_outputs:
                                agent_outputs[role] += "\n\n" + content
                            elif content:
                                agent_outputs[role] = content

                        # Therapist Summary
                        if "TherapistAgent" in agent_outputs:
                            st.subheader("🤗 Compassionate Summary")
                            st.markdown(agent_outputs["TherapistAgent"])

                        # Data Analyst Dashboard
                        if "DataAnalystAgent" in agent_outputs:
                            st.subheader("📈 Mental Health & Insights Dashboard")
                            st.markdown("**Analyzed Metrics:**")
                            dashboard_content = agent_outputs["DataAnalystAgent"]
                            st.markdown(dashboard_content)
                            if "```" in dashboard_content:
                                st.code(dashboard_content, language="text")

                        # Routine Planner
                        if "RoutinePlannerAgent" in agent_outputs:
                            st.subheader("📅 7-Day Self-Care Plan")
                            st.markdown(agent_outputs["RoutinePlannerAgent"])

                        # Fallback
                        if not agent_outputs:
                            st.json(report_data)
                        else:
                            st.caption(f"Generated on: {report_data.get('timestamp', 'Unknown')}")

                        st.success("Report generated and stored successfully.")
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")


# Footer
st.markdown("---")
st.caption("Powered by Zenark Empathy AI | Built with ❤️ for mental health support.")