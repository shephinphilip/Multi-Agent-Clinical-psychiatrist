"""
Zenark - AI Clinical Psychiatry Companion
Streamlit Interface
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os

from zenark_brain import ZenarkBrain
from zenark_db import ZenarkDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Zen - Your Personal Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background-color: rgb(14, 17, 23);
        margin-left: 2rem;
    }
    .zen-message {
        background-color: rgb(14, 17, 23);
        margin-right: 2rem;
    }
    .crisis-alert {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
        .info-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 500;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.username = ""  # Changed from None to ""
        st.session_state.conversation_started = False
        st.session_state.messages = []
        st.session_state.zenark_brain = None
        st.session_state.session_id = None
        st.session_state.is_concluded = False
        st.session_state.show_report = False
        st.session_state.report_data = None
        
        logger.info("✅ Session state initialized")

def get_zenark_brain() -> ZenarkBrain:
    """Get or create Zenark brain instance"""
    if st.session_state.zenark_brain is None:
        st.session_state.zenark_brain = ZenarkBrain()
        logger.info("🧠 Zenark Brain initialized")
    
    return st.session_state.zenark_brain

def display_message(speaker: str, message: str, timestamp: Optional[str] = None):
    """Display a chat message"""
    css_class = "user-message" if speaker == "User" else "zen-message"
    
    with st.container():
        st.markdown(
            f'<div class="chat-message {css_class}">'
            f'<strong>{speaker}:</strong><br>{message}'
            f'</div>',
            unsafe_allow_html=True
        )

def display_conversation_history():
    """Display conversation history"""
    if st.session_state.messages:
        for msg in st.session_state.messages:
            display_message(msg['speaker'], msg['message'])

def start_conversation():
    """Start a new conversation"""
    brain = get_zenark_brain()
    
    response = brain.start_session(st.session_state.username)
    
    st.session_state.conversation_started = True
    st.session_state.session_id = response['session_id']
    st.session_state.messages.append({
        'speaker': 'Zen',
        'message': response['message']
    })
    
    logger.info(f"✅ Conversation started for {st.session_state.username}")

def process_user_input(user_input: str):
    """Process user input and get response"""
    if not user_input.strip():
        return
    
    # Add user message to history
    st.session_state.messages.append({
        'speaker': 'User',
        'message': user_input
    })
    
    # Get brain response
    brain = get_zenark_brain()
    
    try:
        response = brain.process_user_input(user_input)
        
        if 'error' in response:
            st.error(f"Error: {response['error']}")
            return
        
        # Add Zen's response to history
        st.session_state.messages.append({
            'speaker': 'Zen',
            'message': response['message']
        })
        
        # Check if concluded
        if response.get('is_concluded'):
            st.session_state.is_concluded = True
            st.session_state.report_data = response.get('report')
        
        # Check for crisis
        if response.get('is_crisis'):
            st.warning("🆘 Crisis situation detected. Please see the resources below.")
        
        logger.info(f"✅ Processed input. Phase: {response.get('phase')}")
        
    except Exception as e:
        logger.error(f"❌ Error processing input: {e}")
        st.error(f"An error occurred: {str(e)}")

def display_sidebar():
    """Display sidebar with info and controls"""
    with st.sidebar:
        st.markdown("### 💙 About Zen")
        st.info("""
        Zen is your personal companion - a warm, empathetic AI designed to listen and understand how you're feeling.
        
        This is a safe, confidential space for conversation.
        """)
        
        if st.session_state.conversation_started:
            st.markdown("---")
            st.markdown("### 📊 Session Info")
            
            brain = get_zenark_brain()
            session_state = brain.get_session_state()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions", session_state.get('questions_asked', 0))
            with col2:
                confidence = session_state.get('confidence_score', 0.0)
                st.metric("Confidence", f"{confidence*100:.0f}%")
            
            # Show category scores if available
            if session_state.get('category_scores'):
                st.markdown("#### Current Scores")
                for category, score in session_state['category_scores'].items():
                    if score > 0:  # Only show categories with scores
                        st.progress(min(score/3.0, 1.0), text=f"{category.title()}: {score:.2f}")
            
            st.markdown("---")
            
            # Conclude button
            if not st.session_state.is_concluded:
                if st.button("🏁 Conclude Assessment", type="secondary"):
                    brain = get_zenark_brain()
                    response = brain.force_conclude()
                    
                    st.session_state.messages.append({
                        'speaker': 'Zen',
                        'message': response['message']
                    })
                    
                    if response.get('can_conclude', True):
                        st.session_state.is_concluded = True
                        st.session_state.report_data = response.get('report')
                        st.rerun()
            
            # Reset conversation
            st.markdown("---")
            if st.button("🔄 Start New Conversation", type="secondary"):
                # Reset conversation state
                st.session_state.conversation_started = False
                st.session_state.messages = []
                st.session_state.is_concluded = False
                st.session_state.show_report = False
                st.session_state.report_data = None
                st.session_state.zenark_brain = None
                st.rerun()
            
            # Delete account
            if st.button("🗑️ Delete My Data", type="secondary"):
                db = ZenarkDB()
                if db.delete_user(st.session_state.username):
                    st.success("✅ Your data has been deleted")
                    st.session_state.username = None
                    st.session_state.conversation_started = False
                    st.session_state.messages = []
                    st.session_state.zenark_brain = None
                    st.rerun()
        
        # Crisis resources (always visible)
        st.markdown("---")
        st.markdown("### 🆘 Crisis Resources")
        st.error("""
        **If you're in immediate danger:**
        - **988**: Suicide & Crisis Lifeline
        - **741741**: Crisis Text Line (text HOME)
        - **911**: Emergency Services
        
        You're not alone. Help is available 24/7.
        """)

def display_report():
    """Display assessment report"""
    if not st.session_state.report_data:
        return
    
    report = st.session_state.report_data
    
    st.markdown("---")
    st.markdown("## 📋 Assessment Summary")
    
    # Key findings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric(
            "Primary Area",
            report.get('primary_category', 'N/A').replace('_', ' ').title()
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric(
            "Severity",
            report.get('severity', 'N/A').title()
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric(
            "Confidence",
            f"{report.get('confidence', 0)*100:.0f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Category scores
    if report.get('category_scores'):
        st.markdown("### 📊 Category Breakdown")
        
        for category, score in sorted(
            report['category_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score > 0:  # Only show categories with scores
                st.progress(
                    min(score/3.0, 1.0), 
                    text=f"{category.replace('_', ' ').title()}: {score:.2f}"
                )
    
    # Instruments used
    if report.get('instruments_used'):
        st.markdown("### 🔬 Assessment Tools Used")
        instruments = report['instruments_used']
        num_cols = min(len(instruments), 4)
        cols = st.columns(num_cols)
        
        for col, (instrument, count) in zip(cols, instruments.items()):
            with col:
                st.info(f"**{instrument}**\n\n{count} questions")
    
    # Recommendations
    if report.get('recommendations'):
        st.markdown("### 💡 Recommendations")
        st.markdown(
            f'<div class="info-box">{report["recommendations"]}</div>',
            unsafe_allow_html=True
        )
    
    # Important disclaimer
    st.warning("""
    **Important:** This assessment is a screening tool, not a diagnosis. 
    Please consult with a qualified mental health professional for proper evaluation and treatment.
    """)
    
    # Download report button
    if st.button("📥 Download Report", type="primary"):
        import json
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download as JSON",
            data=report_json,
            file_name=f"zenark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧠 Zen</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Your Personal Companion</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    display_sidebar()
    
    # Main content area
    if not st.session_state.username:
        # Login/Welcome screen
        st.markdown("### Welcome! 👋")
        st.info("""
        Zen is here to listen and support you. This is a safe, confidential space where you can share openly.
        
        To get started, please enter your name below.
        """)
        
        with st.form("login_form"):
            username = st.text_input("What should I call you?", placeholder="Enter your name")
            submitted = st.form_submit_button("Start Conversation", type="primary")
            
            if submitted and username:
                st.session_state.username = username.strip()
                st.rerun()
    
    elif not st.session_state.conversation_started:
        # Start conversation
        start_conversation()
        st.rerun()
    
    else:
        # Main conversation interface
        
        # Display conversation history
        display_conversation_history()
        
        # Show report if concluded
        if st.session_state.is_concluded and st.session_state.report_data:
            display_report()
        
        # Input area (only if not concluded)
        if not st.session_state.is_concluded:
            with st.form("input_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Your message:",
                    placeholder="Type your message here...",
                    height=100,
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([4, 1])
                with col2:
                    submitted = st.form_submit_button("Send 📤", type="primary", use_container_width=True)
                
                if submitted and user_input:
                    process_user_input(user_input)
                    st.rerun()
        else:
            st.info("✅ Assessment complete. You can start a new conversation from the sidebar if you'd like to talk more.")

if __name__ == "__main__":
    main()