import streamlit as st
from pymongo import MongoClient
import datetime, os, json, random
from dotenv import load_dotenv

# Import backend functions directly
from Zenark_Empathy import generate_response, save_conversation, health_check

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

# Maintain session state
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
        marks = st.number_input(
            f"{sub} (out of 100)",
            min_value=0,
            max_value=100,
            step=1,
            value=random.randint(60, 100),
            key=sub,
        )
        marks_data.append({"subject": sub, "marks": marks})

    new_subject = st.text_input("Optional Subject")
    new_marks = st.number_input(
        "Marks (out of 100):",
        min_value=0,
        max_value=100,
        step=1,
        value=random.randint(60, 100),
        key="custom",
    )
    if new_subject:
        marks_data.append({"subject": new_subject, "marks": new_marks})

    if st.button("Submit"):
        if not name:
            st.error("Please enter your name before submitting.")
        else:
            record = {"name": name, "marks": marks_data, "timestamp": datetime.datetime.now()}
            marks_col.insert_one(record)
            st.session_state["student_name"] = name
            st.success("Marks stored successfully.")
            st.session_state["current_page"] = "Empathy Chatbot"
            st.rerun()

# ======================================================================
# 2. EMPATHY CHATBOT (CALLS generate_response)
# ======================================================================
elif page == "Empathy Chatbot":
    st.title("🧠 Zenark Empathy Chatbot")

    max_q = st.sidebar.selectbox("Maximum Questions", [5, 10, 15, 20], index=1)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        st.session_state.q_count = 0
        st.session_state.finished = False
        st.session_state.name = st.session_state.get("student_name", "User")
        st.session_state.initialized = False

    # First message
    if not st.session_state.get("initialized"):
        first_msg = (
            f"Hi {st.session_state.name}. How are you feeling today? "
            "Is there anything on your mind that you'd like to talk about?"
        )
        st.session_state.conversation.append({"ai": first_msg})
        st.session_state.initialized = True
        st.rerun()

    # Display chat
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

        # ✅ Directly call backend logic
        ai_reply = generate_response(user_input, st.session_state.name, st.session_state.q_count, max_q)


        st.session_state.conversation.append({"ai": ai_reply})
        st.chat_message("assistant").write(ai_reply)

        # Save when done
        if st.session_state.q_count >= max_q:
            st.session_state.finished = True
            goodbye = "That's all for now. Thank you for sharing — take care."
            st.session_state.conversation.append({"ai": goodbye})
            st.chat_message("assistant").write(goodbye)

            save_conversation(st.session_state.conversation, st.session_state.name)
            st.success("Conversation saved successfully.")

            # Download
            json_str = json.dumps(
                {
                    "name": st.session_state.name,
                    "conversation": st.session_state.conversation,
                    "timestamp": str(datetime.datetime.now()),
                },
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                label="⬇️ Download Conversation (JSON)",
                data=json_str,
                file_name=f"zenark_conversation_{st.session_state.name}.json",
                mime="application/json",
            )

# ======================================================================
# 3. REPORT GENERATION
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
        st.text_area("Last Conversation", conversation_text, height=250)
        st.info("Generating summary...")

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            summary = llm.invoke([HumanMessage(content=f"Summarize this empathetically:\n{conversation_text}")])
            report = summary.content
            reports_col.insert_one({"name": last_chat["name"], "report": report, "timestamp": datetime.datetime.now()})
            st.success("Report generated successfully.")
            st.write(report)
        except Exception as e:
            st.error(f"Report generation failed: {e}")
