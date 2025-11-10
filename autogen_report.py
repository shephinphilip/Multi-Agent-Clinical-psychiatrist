from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os, json, datetime
from dotenv import load_dotenv
from Guideliness import action_scoring_guidelines

# Load environment
load_dotenv()
os.environ["AUTOGEN_USE_DOCKER"] = "0"


# -------------------------------------------
# ZENARK INTERVENTION SPECIALIST ROLE
# -------------------------------------------
ZENARK_INTERVENTION_GUIDELINES = """
ZENARK AGENT ROLE: INTERVENTION SPECIALIST (v2.0)
Confidential Clinical Guidelines for Generating Student Wellness Guides

Your function: analyze a student's session summary (primary stressor, emotion, and Global Distress Score) 
and output ONE formatted text block titled "Your Personal Wellness Guide".

Your structure:
PART 1: Validation of the user's current struggle
PART 2: One immediate micro-action
PART 3: ZenMode prescription
PART 4: Professional safety net

Follow the Intervention Matrix (A-E) and Crisis Protocols (GDS 9-10) strictly. 
Never output more than one guide. 
Always speak in warm, validating, non-judgmental tone.
"""


def generate_autogen_report(conversation_text: str, name: str):
    """Generate a 3-part reflective report:
       1. Personalized Wellness Guide (Intervention Specialist)
       2. Strengths / Weaknesses Dashboard (Data Analyst)
       3. 7-Day Self-Care Plan (Routine Planner)
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm_cfg = {
        "model": "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "temperature": 0.7,
    }

    # 1️⃣ Wellness Guide
    intervention_agent = AssistantAgent(
        name="InterventionSpecialistAgent",
        llm_config=llm_cfg,
        system_message=(
            f"You are the Zenark Intervention Specialist.\n{ZENARK_INTERVENTION_GUIDELINES}\n\n"
            "Input: conversation summary.\n"
            "Output: One warm, empathetic block titled 'Your Personal Wellness Guide'."
        ),
    )

    # 2️⃣ Strengths / Weaknesses Analyst
    data_analyst = AssistantAgent(
        name="DataAnalystAgent",
        llm_config=llm_cfg,
        system_message=(
            "You are the Strengths–Weaknesses Analyst.\n"
            "Your role is to extract clear, behaviorally worded strengths and weaknesses "
            "from the student's overall conversation.\n\n"
            "Output must be a valid JSON object in this format:\n"
            "{\n"
            '  "strengths": ["...","..."],\n'
            '  "weaknesses": ["...","..."]\n'
            "}\n\n"
            "Guidelines:\n"
            "• Identify emotional, social, or behavioral strengths (e.g., resilience, openness, self-awareness).\n"
            "• Identify weaknesses as skill gaps or current challenges (e.g., procrastination, self-doubt, poor focus).\n"
            "• Avoid generic adjectives; use specific, observable phrasing.\n"
            "• After the JSON, include a 2–3 sentence text summary that captures the student's overall pattern."
        ),
    )


    # 3️⃣ 7-Day Planner
    planner = AssistantAgent(
        name="RoutinePlannerAgent",
        llm_config=llm_cfg,
        system_message=(
            "Design a 7-day self-care plan that includes:\n"
            "1. Daily activities for wellbeing and social connection.\n"
            "2. One 'Strength + Growth Focus' block.\n"
            "3. Zen Mode recommendation each day.\n"
            "Keep it actionable, realistic, and culturally contextualized for Indian students."
        ),
    )

    # Group Chat Orchestration
    mgr = GroupChatManager(
        groupchat=GroupChat(
            agents=[intervention_agent, data_analyst, planner],
            messages=[],
            max_round=3,
        ),
        llm_config=llm_cfg,
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        system_message="Provide the conversation text and request a 3-part wellness report.",
        llm_config=llm_cfg,
    )

    prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        "Generate:\n"
        "1. 'Your Personal Wellness Guide'\n"
        "2. Strengths & Weaknesses JSON and dashboard summary\n"
        "3. 7-Day Self-Care Plan\n"
        "Each part must be clearly delineated with headings."
    )

    user_proxy.initiate_chat(mgr, message=prompt)

    messages = mgr.groupchat.messages
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse strengths/weaknesses
    extracted = {"strengths": [], "weaknesses": []}
    for msg in messages:
        txt = str(msg.get("content", ""))
        try:
            if "strength" in txt.lower() or "weakness" in txt.lower():
                obj = json.loads(txt)
                extracted["strengths"].extend(obj.get("strengths", []))
                extracted["weaknesses"].extend(obj.get("weaknesses", []))
        except Exception:
            continue

    # ===================== FINAL REPORT AGGREGATION =====================
    # Collect structured agent outputs in a predictable list
    report_json = []
    for m in mgr.groupchat.messages:
        if isinstance(m, dict):
            report_json.append(m)
        elif hasattr(m, "content") and isinstance(m.content, dict):
            report_json.append(m.content)

    # Fallback: remove duplicates or None
    report_json = [r for r in report_json if r]

    # Create a single unified report object
    report_data = {
        "name": name,
        "timestamp": timestamp,
        "report": report_json
    }

    return report_data
