import streamlit as st
import json, os, random, uuid
from datetime import datetime
from typing import List, Dict, Optional, cast
from pymongo import MongoClient
from dotenv import load_dotenv

# Import backend functions directly
from zenarck_langraph import generate_response, save_conversation, generate_report

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, maxPoolSize=5)
db = client["zenark_db"]
marks_col = db["student_marks"]
reports_col = db["reports"]

# Page setup
st.set_page_config(page_title="Zenark Empathy Chat", page_icon="💬", layout="wide")

# Style
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

# ----------------------- MARK ENTRY -----------------------
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

# ----------------------- CHAT TAB -----------------------
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
        # Display history
        for msg in st.session_state.conversation:
            with st.chat_message("user" if "user" in msg else "assistant"):
                # Clean AI messages to remove source citations before display
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
                    # Clean the reply to remove source before display and save
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
                st.success("Conversation saved.")
                st.json(record)
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation = []
                st.session_state.question_index = 1
                st.rerun()

# ----------------------- REPORT TAB -----------------------
with report_tab:
    st.header("📊 Generate Empathy Report")
    name_input = st.text_input("Enter name", value=st.session_state.name or "")

    if st.button("Generate Report"):
        if not name_input:
            st.warning("Enter a name.")
        else:
            with st.spinner("Generating reflective report..."):
                report_data = generate_report(name_input.strip())

                if "error" in report_data:
                    st.error(report_data["error"])
                else:
                    st.success("Report generated successfully.")
                    messages = report_data.get("report", [])
                    agent_outputs = {}
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("name", "Unknown")
                            content = str(msg.get("content", "")).strip()
                            if content:
                                agent_outputs.setdefault(role, "")
                                agent_outputs[role] += "\n\n" + content


                    if "TherapistAgent" in agent_outputs:
                        st.subheader("🤗 Compassionate Summary")
                        st.markdown(agent_outputs["TherapistAgent"])

                    if "DataAnalystAgent" in agent_outputs:
                        st.subheader("📈 Insights Dashboard")
                        st.markdown(agent_outputs["DataAnalystAgent"])

                    if "RoutinePlannerAgent" in agent_outputs:
                        st.subheader("📅 7-Day Self-Care Plan")
                        st.markdown(agent_outputs["RoutinePlannerAgent"])

                        # Fallback
                        if not agent_outputs:
                            st.json(report_data)
                        else:
                            st.caption(f"Generated on: {report_data.get('timestamp', 'Unknown')}")

                        st.success("Report generated and stored successfully.")
        


# Footer
st.markdown("---")
st.caption("Powered by Zenark Empathy AI")