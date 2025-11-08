# autogen_report.py
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os, json, datetime
from dotenv import load_dotenv
from Guideliness import action_scoring_guidelines

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
    data_analyst = AssistantAgent(
        name="DataAnalystAgent",
        llm_config=llm_cfg,
        system_message=(
            "Analyze the conversation or provided data for insights related to the user's mental health, strengths, and weaknesses. "
            f"Extract relevant metrics using this guidelines {action_scoring_guidelines} and present them in a diagrammatic dashboard format using text-based visualizations, such as pie chart , bar graph."
        ),
    )
    planner = AssistantAgent(
        name="RoutinePlannerAgent",
        llm_config=llm_cfg,
        system_message=(
            "Design a 7‑day self‑care plan that includes: "
            "1) daily activities for wellbeing and social connection, "
            "2) a dedicated 'Strength + Growth Focus' block where the user leverages a personal strength "
            "and works on a specific weakness, "
            "3) a 'Zen Mode' suggestion (guided meditation, breathing exercise, or music) "
            "tailored to the user's current emotional state at the end of the conversation."
        ),
    )


    mgr = GroupChatManager(
        groupchat=GroupChat(agents=[therapist, data_analyst, planner], messages=[], max_round=3),
        llm_config=llm_cfg,
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        system_message="Provide the conversation text and request a full reflective report.",
        llm_config=llm_cfg,
    )

    prompt = f"Conversation:\n{conversation_text}\n\nGenerate a compassionate summary, a diagrammatic dashboard analyzing mental health, strengths, and weaknesses, and a 7-day self-care plan."
    user_proxy.initiate_chat(mgr, message=prompt)

    report_json = mgr.groupchat.messages
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "name": name,
        "timestamp": timestamp,
        "report": report_json,
    }