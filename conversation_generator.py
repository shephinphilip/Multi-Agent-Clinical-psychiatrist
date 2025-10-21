import logging
from typing import Dict, List, Optional
import openai
import random
import os
from openai import OpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

class ConversationGenerator:
    """Generates natural, empathetic conversation responses"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or OPEN_AI_API_KEY)
        self.model = model
        self.greeting_templates = [
            "Hey {name}, I'm really glad you're here. How have you been feeling?",
            "Hi {name}, it's good to see you. What's been on your mind lately?",
            "Hello {name}, thanks for coming to talk with me. How are things going for you?",
        ]
        
        self.goodbye_templates = [
            "Take care of yourself, {name}. Remember, I'm here whenever you need to talk.",
            "Thank you for sharing with me today, {name}. You've shown real courage.",
            "I'm glad we had this conversation, {name}. Stay strong, and reach out anytime.",
        ]
    
    def generate_greeting(self, username: Optional[str] = None, is_returning: bool = False) -> str:
        """Generate personalized greeting"""
        if is_returning:
            greetings = [
                f"Welcome back, {username}! It's good to see you again. How have things been since we last talked?",
                f"Hi {username}, I'm glad you came back. What's been happening in your world?",
                f"Hey {username}, welcome back! What would you like to talk about today?",
            ]
            return random.choice(greetings)
        else:
            intro = f"""Hey there! I'm Zen, your personal companion. I'm here to listen, understand, and support you through whatever you're experiencing.

This is a safe, judgment-free space where you can share openly. Everything we discuss is confidential.

{f"What would you like me to call you?" if not username else f"Nice to meet you, {username}! How are you feeling today?"}"""
            return intro
    
    def generate_empathetic_response(self, context: Dict) -> str:
        """Generate empathetic acknowledgment of user's feelings"""
        user_message = context.get('user_message', '')
        emotion_analysis = context.get('emotion_analysis', {})
        conversation_history = context.get('conversation_history', [])
        memory_summary = context.get('memory_summary') or "No detailed history captured yet."
        memory_highlight = context.get('memory_highlight') or ""

        system_prompt = """You are Zen, a warm, empathetic companion AI. You're having a natural conversation with someone who may be going through difficult times.

Your communication style:
- Warm and genuine, like a caring friend
- Validate their feelings without being patronizing
- Use natural, conversational language
- Avoid clinical jargon or sounding like a therapist
- Show you're listening by reflecting what they've shared
- Provide 3-4 sentences that acknowledge their situation and offer gentle reassurance
- NEVER start every response with "It sounds like..." - vary your responses naturally

Examples of natural responses:
- "That must be really hard to deal with."
- "I can hear how much that's been weighing on you."
- "Thanks for sharing that with me. That takes courage."
- "I'm here with you. Tell me more about that if you'd like."
- "That sounds really overwhelming right now."
"""
        
        recent_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-3:] if msg.get('role') and msg.get('content')
        ]) if conversation_history else "First interaction"
        user_prompt = f"""Recent conversation:
{recent_history}

User just said: "{user_message}"

Key things to remember from our ongoing chat: {memory_summary}
{f"Specific highlight: {memory_highlight}" if memory_highlight else ""}

Generate a natural empathetic response in 3-4 sentences. Focus on acknowledging their feelings, referencing the memory information when it helps, and offer gentle reassurance without asking any follow-up questions."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=100
            )
            content = response.choices[0].message.content
        
            # Defensive check for None (prevents the "strip" is not a known attribute of "None" error)
            if content is not None:
                return content.strip()
            else:
                # Handle empty response (e.g., return a default string)
                return (
                    "Thank you for sharing that with me. I can hear how much this matters to you, and I'm keeping our earlier conversation in mind as we talk. You're not alone in this space. I'm right here, staying with you while you work through these feelings."
                )
        except Exception as e:
            logger.error(f"❌ Failed to generate empathetic response: {e}")
            return (
                "I hear you, and I know this is heavy. I'm keeping what you've already shared in mind so you don't have to repeat yourself, and I want you to know your feelings are valid. We can take this one step at a time together. I'm staying right here with you."
            )
    
    def refine_clinical_question(self, context: Dict) -> str:
        """Convert clinical question into natural conversation"""
        clinical_question = context.get('clinical_question', {})
        question_text = clinical_question.get('question_text', '') or clinical_question.get('text', '')
        category = clinical_question.get('category', '')
        options = clinical_question.get('options', [])
        username = context.get('username', '')
        conversation_history = context.get('conversation_history', [])
        empathy_level = context.get('empathy_level', 'moderate')
        
        # ✅ Build rich context about what we know
        memory_context = []
        for msg in conversation_history[-5:]:
            if msg.get('role') == 'user' and 'exam' in msg.get('content', '').lower():
                memory_context.append('user is stressed about upcoming math exams')
            if msg.get('role') == 'user' and 'homework' in msg.get('content', '').lower():
                memory_context.append('user struggles with homework completion')
            if msg.get('role') == 'user' and any(word in msg.get('content', '').lower() for word in ['cant sleep', 'worry', 'anxious']):
                memory_context.append('user experiences sleep issues due to exam stress')
            if msg.get('role') == 'user' and 'parents' in msg.get('content', '').lower():
                memory_context.append('user has strict parents')
            if msg.get('role') == 'user' and 'friends' in msg.get('content', '').lower():
                memory_context.append('user values time with friends')
        
        # ✅ VARIETY PROMPT - no more repetition
        system_prompt = f"""You are Zen, a warm and empathetic companion AI. Your communication style is natural, conversational, and deeply attentive.

    STRICT RULES:
    - NEVER use the phrase "It sounds like" 
    - NEVER use "I know things have been going through" or "In the midst of everything"
    - NEVER start with "It sounds like you're going through a lot" or "You seem to be going through a lot"
    - Use a variety of natural openings that fit the context
    - Make it feel like ONE conversation, not an interview
    - Connect each question to the specific thing the user just shared
    - Sound like a caring friend, not a researcher

    CONVERSATION HISTORY:
    {conversation_history[-3:] if conversation_history else 'Starting conversation'}

    BACKGROUND:
    - {' '.join(memory_context) if memory_context else 'User is a student dealing with academic stress'}

    Your personality:
    - Warm, genuine, and perceptive
    - Uses natural conversational flow
    - Builds on what the user said
    - Shows continuity by referencing past topics
    - Varies expressions naturally

    Examples of GOOD natural transitions to clinical topics:
    ✅ "You mentioned not sleeping well because of exam stress. When you're lying awake at night, what kinds of thoughts go through your mind?"
    ✅ "Math seems really overwhelming right now. When you're feeling stuck like this, does it affect how you see yourself or your abilities?"
    ✅ "With your parents being so strict, do you ever feel like you can't be honest about how hard things are?"
    ✅ "You said you can't enjoy cartoons anymore. What does it feel like when you try to relax but can't?"
    ✅ "You mentioned feeling alone at school. Can you tell me more about what that loneliness feels like?"
    ✅ "When you think about your future and feel unsure, what's the biggest concern that comes up?"

    ALWAYS:
    - Connect to what was just said
    - Use the user's words
    - Be specific, not generic
    - Never use the same opening twice
    - Sound organic and spontaneous
    - Make it flow naturally from the previous sentence

    NEVER:
    - Ask robotic checklist questions
    - Use clinical language
    - Sound like you're following a script
    - Repeat phrases
    - Ask multiple questions at once
    - Make it feel like an assessment"""
        
        recent_exchange = ""
        if conversation_history:
            last_messages = conversation_history[-3:]
            recent_exchange = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in last_messages if msg.get('role') and msg.get('content')
            ])
        
        user_prompt = f"""RECENT CONVERSATION:
    {recent_exchange if recent_exchange else 'First interaction'}

    CLINICAL QUESTION TO INTEGRATE: "{question_text}"
    CATEGORY: {category}
    ANSWER OPTIONS: {', '.join(options)}

    INSTRUCTION: Re-phrase this clinical question as a natural, contextual follow-up that flows from the conversation. Use the examples above as inspiration for variety and naturalness.

    RESPONSE: (1-2 sentences max)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            natural_question = response.choices[0].message.content
            
            # Add options if it's a multiple choice
            if natural_question is not None:
                return natural_question.strip()
            else:
                # Handle empty response (e.g., return a default string)
                return "Could you tell me more about yourself?"
            
        except Exception as e:
            logger.error(f"❌ Failed to refine question: {e}")
            return question_text
    
    def _format_options(self, options: List[str]) -> str:
        """Format answer options naturally"""
        if len(options) <= 4:
            return " | ".join(options)
        else:
            formatted = "Options:\n"
            for i, option in enumerate(options, 1):
                formatted += f"{i}. {option}\n"
            return formatted.strip()
    
    def generate_transition(self, context: Dict) -> str:
        """Generate smooth transitions between topics"""
        from_topic = context.get('from_topic', '')
        to_topic = context.get('to_topic', '')
        conversation_history = context.get('conversation_history', [])
        
        # Get recent conversation exchange
        recent_exchange = ""
        if conversation_history:
            last_messages = conversation_history[-2:]
            recent_exchange = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in last_messages if msg.get('role') and msg.get('content')
            ])
        
        system_prompt = """You are Zen, creating smooth conversational transitions. 
        Connect topics naturally without being abrupt.
        Keep it brief - one sentence that bridges the topics."""
        
        # ✅ FIX: Include recent_exchange in the prompt
        user_prompt = f"""Recent conversation:
    {recent_exchange if recent_exchange else 'Initial conversation'}

    Create a natural transition from discussing "{from_topic}" to asking about "{to_topic}".

    Example transitions:
    "I appreciate you sharing that. Can I ask you about something else?"
    "That makes sense. I'm also curious about..."
    "Thanks for telling me. There's something else I'd like to understand..."

    Generate one natural transition sentence:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            content = response.choices[0].message.content
        
            if content is not None:
                return content.strip()
            else:
                return "Could you tell me more about yourself?"
        except:
            return "I appreciate you sharing that. Can I ask about something else?"
    
    def generate_casual_conversation(self, context: Dict) -> str:
        """Generate casual conversation for users who seem well"""
        username = context.get('username', '')
        conversation_history = context.get('conversation_history', [])
        topics_discussed = context.get('topics_discussed', [])
        
        # Topics to explore for general wellness
        casual_topics = [
            'school_academics',
            'friendships',
            'hobbies',
            'family',
            'future_goals',
            'recent_achievements',
            'challenges',
            'stress_management'
        ]
        
        # Filter out already discussed topics
        available_topics = [t for t in casual_topics if t not in topics_discussed]
        
        if not available_topics:
            return self.generate_positive_conclusion(context)
        
        next_topic = available_topics[0]
        
        system_prompt = f"""You are Zen, having a casual, supportive conversation with {username} who seems to be doing well overall.

Ask naturally about: {next_topic.replace('_', ' ')}

Style:
- Conversational and friendly
- Show genuine interest
- Keep it light but meaningful
- 1-2 sentences max
- Connect to previous conversation if relevant

Examples:
"So tell me, what's a typical day like for you at school?"
"What do you like to do for fun? Any hobbies or activities you're into?"
"How are things with your friends? Anyone you're particularly close with?"
"""

        recent_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-3:] if msg.get('role') and msg.get('content')
        ]) if conversation_history else ""

        user_prompt = f"""Recent conversation:
{recent_history}

Generate a natural question about: {next_topic.replace('_', ' ')}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=100
            )
            content = response.choices[0].message.content
        
            # Defensive check for None (prevents the "strip" is not a known attribute of "None" error)
            if content is not None:
                return content.strip()
            else:
                # Handle empty response (e.g., return a default string)
                return "Could you tell me more about yourself?"
        except Exception as e:
            logger.error(f"❌ Failed to generate casual conversation: {e}")
            return "What's something you've been enjoying lately?"
    
    def generate_positive_conclusion(self, context: Dict) -> str:
        """Generate conclusion for users who are doing well"""
        username = context.get('username', '')
        conversation_summary = context.get('conversation_summary', '')
        
        system_prompt = f"""You are Zen, concluding a conversation with {username} who appears to be doing well mentally and emotionally.

Generate a warm, encouraging conclusion that:
1. Acknowledges the positive conversation
2. Validates their wellness
3. Encourages them to reach out if things change
4. Ends on a supportive note

Keep it genuine and warm - 2-3 sentences."""

        user_prompt = f"""Conversation summary: {conversation_summary if conversation_summary else 'User seems emotionally stable and well'}

Generate a positive conclusion message:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=150
            )
            content = response.choices[0].message.content
        
            # Defensive check for None (prevents the "strip" is not a known attribute of "None" error)
            if content is not None:
                return content.strip()
            else:
                # Handle empty response (e.g., return a default string)
                return "Could you tell me more about yourself?"
        except Exception as e:
            logger.error(f"❌ Failed to generate conclusion: {e}")
            return f"It's been really nice talking with you, {username}. You seem to be doing well, which is great to see! Remember, I'm here anytime you want to chat or if things feel different. Take care of yourself!"
    
    def generate_assessment_conclusion(self, context: Dict) -> str:
        """Generate conclusion after assessment completion"""
        username = context.get('username', '')
        primary_category = context.get('primary_category', '')
        severity = context.get('severity', '')
        confidence = context.get('confidence', 0.0)
        
        system_prompt = f"""You are Zen, concluding an assessment with {username}.

You've identified potential concerns in the area of: {primary_category}
Severity level: {severity}

Generate a compassionate conclusion that:
1. Thanks them for their openness
2. Acknowledges their courage in sharing
3. Gently explains that you've gathered enough information
4. Encourages them (without being dismissive)
5. Mentions that they'll receive a summary
6. Reminds them professional support is valuable

Tone: Warm, supportive, hopeful
Length: 3-4 sentences
IMPORTANT: DO NOT diagnose. Say you've "noticed some patterns" or "gathered information" - never say "you have X condition"."""

        user_prompt = f"""Generate a compassionate conclusion for someone showing signs of {primary_category} ({severity} level)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            content = response.choices[0].message.content
        
            # Defensive check for None (prevents the "strip" is not a known attribute of "None" error)
            if content is not None:
                return content.strip()
            else:
                # Handle empty response (e.g., return a default string)
                return "Could you tell me more about yourself?"
        except Exception as e:
            logger.error(f"❌ Failed to generate assessment conclusion: {e}")
            return f"Thank you so much for sharing with me today, {username}. You've shown real courage in opening up about how you've been feeling. I've gathered some helpful information from our conversation. Remember, reaching out for professional support can make a real difference. You're taking positive steps, and that's something to be proud of."
    
    def generate_crisis_response(self, context: Dict) -> str:
        """Generate immediate crisis response"""
        username = context.get('username', '')
        specific_concerns = context.get('specific_concerns', [])
        
        crisis_response = f"""I hear you, {username}, and I'm really concerned about what you're sharing with me. Your safety is the most important thing right now.

I need you to know that you don't have to face this alone. Please reach out to someone who can provide immediate support:

 **Crisis Resources:**
• **National Suicide Prevention Lifeline**: 988 (call or text)
• **Crisis Text Line**: Text HOME to 741741
• **Emergency Services**: 911

Please talk to a trusted adult, counselor, or family member right away. These feelings are temporary, and help is available.

Would you like to continue talking, or would you prefer to reach out to one of these resources first?"""

        return crisis_response
    
    def generate_follow_up(self, context: Dict) -> str:
        """Generate natural follow-up based on user's response"""
        user_message = context.get('user_message', '')
        conversation_history = context.get('conversation_history', [])

        memory_summary = context.get('memory_summary') or "No detailed history captured yet."
        memory_highlight = context.get('memory_highlight') or ""

        system_prompt = """You are Zen, responding naturally to what the user just shared.

    Generate a compassionate follow-up that:
    - Shows you're listening
    - Demonstrates you remember details from earlier in the conversation
    - Digs a bit deeper naturally while keeping the focus on their wellbeing
    - Asks an open-ended question that invites them to share more
    - Feels conversational, not interrogative
    - Delivers 3-4 sentences that remain warm and sympathetic"""

        # ✅ FIX: Use 'role' and 'content' instead of 'speaker' and 'message'
        recent_history = "\n".join([
            f"{msg['role']}: {msg['content']}"  # Changed keys here
            for msg in conversation_history[-2:] if msg.get('role') and msg.get('content')
        ]) if conversation_history else "First interaction"

        user_prompt = f"""Conversation:
    {recent_history}

    User just said: "{user_message}"

    Memory summary to reference naturally: {memory_summary}
    {f"Highlight: {memory_highlight}" if memory_highlight else ""}

    Generate a natural follow-up that is 3-4 sentences long, shows empathy, and gently guides them to share more."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=160
            )
            content = response.choices[0].message.content

            if content is not None:
                return content.strip()
            else:
                return (
                    "Thank you for opening up about that. I remember what you mentioned earlier, and I'm holding onto those details so you don't have to repeat yourself. Could you share a bit more about how this is affecting you right now? I'm right here listening with you."
                )

        except Exception as e:
            logger.error(f"❌ Failed to generate follow-up: {e}")
            return (
                "I really appreciate you telling me that. I'm keeping your earlier experiences in mind, and I want to understand this part of the story better. Would you feel comfortable sharing more about what's happening or how it's been affecting you? I'm right here with you."
            )
        
