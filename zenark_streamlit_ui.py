import streamlit as st
from pymongo import MongoClient
import os, json, datetime, requests
from dotenv import load_dotenv
from autogen_report import generate_autogen_report
import random

# ========== ENVIRONMENT ==========
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")  # default local server

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
        # generate random marks between 60 and 100
        random_marks = random.randint(60, 100)
        marks = st.number_input(
            f"{sub} (out of 100)",
            min_value=0,
            max_value=100,
            step=1,
            key=sub,
            value=random_marks,          # 👈 prefilled random number
        )
        marks_data.append({"subject": sub, "marks": marks})

    st.write("Add optional subject:")
    new_subject = st.text_input("Subject name")
    new_marks = st.number_input(
        "Marks (out of 100):",
        min_value=0,
        max_value=100,
        step=1,
        key="custom",
        value=random.randint(60, 100)   # 👈 also random for optional field
    )
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
# 2. EMPATHY CHATBOT — NOW CONNECTED TO FASTAPI
# ======================================================================
elif page == "Empathy Chatbot":
    st.title("🧠 Zenark Empathy Chatbot (API Connected)")

    max_q = st.sidebar.selectbox("Maximum Questions", [5, 10, 15, 20], index=1)
    st.sidebar.info("Select how many questions you want to continue.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        st.session_state.q_count = 0
        st.session_state.finished = False
        st.session_state.name = st.session_state.get("student_name", "User")
        st.session_state.initialized = False

    if not st.session_state.get("initialized"):
        first_msg = (
            f"Hi {st.session_state.name}. How are you feeling today? "
            "Is there anything on your mind that you'd like to talk about?"
        )
        st.session_state.conversation.append({"ai": first_msg})
        st.session_state.initialized = True
        st.rerun()

    # Display chat history
    for turn in st.session_state.conversation:
        if "user" in turn:
            st.chat_message("user").write(turn["user"])
        if "ai" in turn:
            st.chat_message("assistant").write(turn["ai"])

    user_input = st.chat_input("Your response...")

    if user_input and not st.session_state.finished:
        st.session_state.conversation.append({"user": user_input})
        st.chat_message("user").write(user_input)
        st.session_state.q_count += 1

        # Send to FastAPI backend
        try:
            payload = {
                "text": user_input,
                "name": st.session_state.name,
                "question_index": st.session_state.q_count + 1,
                "max_questions": max_q
            }
            response = requests.post(f"{FASTAPI_URL}/chat", json=payload, timeout=90)
            if response.status_code == 200:
                ai_reply = response.json().get("response", "(no response)")
            else:
                ai_reply = f"(Error {response.status_code}: {response.text})"
        except Exception as e:
            ai_reply = f"(Engine error: {e})"

        st.session_state.conversation.append({"ai": ai_reply})
        st.chat_message("assistant").write(ai_reply)

        # Finish and save conversation
        if st.session_state.q_count >= max_q:
            st.session_state.finished = True
            goodbye = "That's all for now. Thank you for sharing — take care."
            st.session_state.conversation.append({"ai": goodbye})
            st.chat_message("assistant").write(goodbye)

            # Save conversation to backend
            try:
                save_payload = {
                    "conversation": st.session_state.conversation,
                    "name": st.session_state.name
                }
                save_res = requests.post(f"{FASTAPI_URL}/save_chat", json=save_payload, timeout=60)
                if save_res.status_code == 200:
                    st.success("Conversation saved successfully via API.")
                else:
                    st.error(f"Save failed: {save_res.status_code}")
            except Exception as e:
                st.error(f"Failed to save conversation: {e}")

            # JSON download
            json_data = {
                "name": st.session_state.name,
                "conversation": st.session_state.conversation,
                "timestamp": str(datetime.datetime.now())
            }
            json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download Conversation (JSON)",
                data=json_str,
                file_name=f"zenark_conversation_{st.session_state.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

            st.info("You can now download your session or go to AutoGen Report.")


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
