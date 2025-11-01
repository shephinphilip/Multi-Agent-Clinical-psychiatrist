import streamlit as st
from pymongo import MongoClient
import os, json, datetime
from dotenv import load_dotenv
from autogen_report import generate_autogen_report
from Zenark_Empathy import generate_response, save_conversation

# ========== ENVIRONMENT ==========
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["zenark_db"]
marks_col = db["student_marks"]
chats_col = db["chat_sessions"]
reports_col = db["reports"]

st.set_page_config(page_title="Zenark Empathy", page_icon="🧠", layout="wide")
st.sidebar.title("Zenark Navigation")
page = st.sidebar.radio("Go to:", ["Marks Entry", "Empathy Chatbot", "AutoGen Report"])

# Maintain current page in session_state for manual navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = page
else:
    page = st.session_state["current_page"]


# ======================================================================
# 1. MARKS ENTRY PAGE
# ======================================================================
if page == "Marks Entry":
    st.title("🎓 Student Marks Submission")
    name = st.text_input("Enter your name:")

    st.write("Enter marks for subjects:")
    default_subjects = ["Maths", "Chemistry", "Physics", "English", "Hindi"]
    marks_data = []

    for sub in default_subjects:
        marks = st.number_input(f"{sub} (out of 100)", min_value=0, max_value=100, step=1, key=sub)
        marks_data.append({"subject": sub, "marks": marks})

    st.write("Add optional subject:")
    new_subject = st.text_input("Subject name")
    new_marks = st.number_input("Marks (out of 100):", min_value=0, max_value=100, step=1, key="custom")
    if new_subject:
        marks_data.append({"subject": new_subject, "marks": new_marks})

    if st.button("Submit"):
        if not name:
            st.error("Please enter your name before submitting.")
        elif any(m["marks"] > 100 for m in marks_data):
            st.error("All marks must be out of 100.")
        else:
            record = {"name": name, "marks": marks_data, "timestamp": datetime.datetime.now()}
            marks_col.insert_one(record)
            st.session_state["student_name"] = name
            st.success("Marks stored successfully.")
            st.session_state["current_page"] = "Empathy Chatbot"
            st.rerun()


# ======================================================================
# 2. EMPATHY CHATBOT (USES Zenark_Empathy.py)
# ======================================================================
elif page == "Empathy Chatbot":
    st.title("🧠 Zenark Empathy Chatbot (LangChain Engine)")

    # Sidebar configuration
    max_q = st.sidebar.selectbox("Maximum Questions", [5, 10, 15, 20], index=1)
    st.sidebar.info("Select how many questions you want to continue.")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        st.session_state.q_count = 0
        st.session_state.finished = False
        st.session_state.name = st.session_state.get("student_name", "User")

        # First generic AI message
        first_msg = f"Hi {st.session_state.name}. What's your sweet name?"
        st.session_state.conversation.append({"ai": first_msg})


    # Display chat history
    for turn in st.session_state.conversation:
        if "user" in turn:
            st.chat_message("user").write(turn["user"])
        if "ai" in turn:
            st.chat_message("assistant").write(turn["ai"])

    # Get user input
    user_input = st.chat_input("Your response...")

    if user_input and not st.session_state.finished:
        st.session_state.conversation.append({"user": user_input})
        st.session_state.q_count += 1

        # Generate AI response (LangChain backend)
        try:
            ai_reply = generate_response(user_input, st.session_state.name)
        except Exception as e:
            ai_reply = f"(Engine error: {str(e)})"

        st.session_state.conversation.append({"ai": ai_reply})
        st.chat_message("assistant").write(ai_reply)

        # End conversation if reached max
        if st.session_state.q_count >= max_q:
            st.session_state.finished = True
            goodbye = "That’s all for now. Thank you for sharing — take care."
            st.chat_message("assistant").write(goodbye)
            st.session_state.conversation.append({"ai": goodbye})

            # Save conversation
            save_conversation(st.session_state.conversation, st.session_state.name)
            st.success("Conversation saved. Proceed to Report page.")
            st.session_state["current_page"] = "AutoGen Report"
            st.rerun()


# ======================================================================
# 3. REPORT GENERATION (AutoGen)
# ======================================================================
elif page == "AutoGen Report":
    st.title("📄 AutoGen Report Generator")

    last_chat = chats_col.find_one(sort=[("_id", -1)])
    if not last_chat:
        st.warning("No previous conversation found.")
    else:
        conversation_text = "\n".join(
            [f"You: {t.get('user','')}\nAI: {t.get('ai','')}" for t in last_chat["conversation"]]
        )

        st.info("Generating report via AutoGen agents...")
        try:
            report_data = generate_autogen_report(conversation_text, last_chat["name"])
            reports_col.insert_one(report_data)
            st.success("Report successfully generated and stored.")
            st.json(report_data["report"])
        except Exception as e:
            st.error(f"Report generation failed: {e}")
