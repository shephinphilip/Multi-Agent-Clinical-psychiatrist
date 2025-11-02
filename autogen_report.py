# autogen_report.py
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os, json, datetime
from dotenv import load_dotenv

load_dotenv()
os.environ["AUTOGEN_USE_DOCKER"] = "0" 

def generate_autogen_report(conversation_text: str, name: str):
    """Generate a multi-agent reflective report from conversation text."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm_cfg = {
        "model": "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "temperature": 0.7,
    }

    therapist = AssistantAgent(
        name="TherapistAgent",
        llm_config=llm_cfg,
        system_message=(
            "Empathetic therapist that summarizes conversations reflectively and compassionately."
        ),
    )
    closure = AssistantAgent(
        name="ClosureAgent",
        llm_config=llm_cfg,
        system_message="Write unsent emotional letters for healing and closure.",
    )
    planner = AssistantAgent(
        name="RoutinePlannerAgent",
        llm_config=llm_cfg,
        system_message="Design a 7-day plan promoting self-care and social connection.",
    )

    mgr = GroupChatManager(
        groupchat=GroupChat(agents=[therapist, closure, planner], messages=[], max_round=3),
        llm_config=llm_cfg,
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        system_message="Provide the conversation text and request a full reflective report.",
        llm_config=llm_cfg,
    )

    prompt = f"Conversation:\n{conversation_text}\n\nGenerate summary, closure letter, and 7-day plan."
    user_proxy.initiate_chat(mgr, message=prompt)

    report_json = mgr.groupchat.messages
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "name": name,
        "timestamp": timestamp,
        "report": report_json,
    }
