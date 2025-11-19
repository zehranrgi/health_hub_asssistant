"""
CVS HealthHub AI - Streamlit Frontend
Interactive chat interface for healthcare agent
"""
import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.healthhub_agent import chat, vector_store

# Page configuration
st.set_page_config(
    page_title="CVS HealthHub AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #0a0e27 !important;
        color: #e0e0e0 !important;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FF4444 0%, #FF8888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .stChatMessage {
        background-color: #1a1f3a !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #e0e0e0 !important;
    }
    [data-testid="chatAvatarIcon-assistant"],
    [data-testid="chatAvatarIcon-user"] {
        background-color: transparent !important;
    }
    .chat-box-assistant {
        background: linear-gradient(135deg, #1a2340 0%, #16213e 100%);
        color: #e0e0e0;
        padding: 20px;
        border-radius: 15px;
        margin: 15px auto;
        max-width: 800px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        border-left: 5px solid #FF6B6B;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .chat-box-user {
        background: linear-gradient(135deg, #1f3a52 0%, #1a3a52 100%);
        color: #e0e0e0;
        padding: 20px;
        border-radius: 15px;
        margin: 15px auto;
        max-width: 800px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        border-right: 5px solid #64B5F6;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏥 CVS HealthHub AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent healthcare information companion</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **CVS HealthHub AI** is an intelligent agent that helps with:

    - 💊 Medication information
    - ⚕️ Drug interaction checks
    - 💉 Vaccine scheduling
    - 🏪 CVS services & hours
    - 💳 Insurance coverage

    *This is informational only - always consult healthcare providers for medical decisions.*
    """)

    st.header("Quick Actions")
    example_queries = [
        "What vaccines are available?",
        "Tell me about blood pressure medications",
        "Check drug interactions for Lisinopril and Aspirin",
        "What are CVS pharmacy hours?",
        "Is my insurance accepted?"
    ]

    st.write("Try these questions:")
    for query in example_queries:
        if st.button(query, key=query):
            st.session_state.selected_query = query

    st.divider()

    # Vector store stats
    try:
        vs_stats = vector_store.get_collection_stats()
        st.header("System Status")
        st.success(f" Knowledge Base: {vs_stats['total_chunks']} documents")
    except:
        st.warning("Knowledge base not loaded")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": """👋 Welcome to CVS HealthHub AI!

I can help you with:
- Medication information and side effects
- Drug interaction checks
- Vaccine availability and scheduling
- CVS pharmacy services and hours
- Insurance coverage questions

What would you like to know?"""
    })

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"""
        <div class="chat-box-assistant">
        {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-box-user">
        {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

# Handle selected query from sidebar
if "selected_query" in st.session_state:
    user_input = st.session_state.selected_query
    del st.session_state.selected_query
else:
    user_input = st.chat_input("Ask about medications, vaccines, services, or insurance...")

# Process user input
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.markdown(f"""
    <div class="chat-box-user">
    {user_input}
    </div>
    """, unsafe_allow_html=True)

    # Get agent response
    with st.spinner("🔍 Searching knowledge base and analyzing..."):
        try:
            result = chat(user_input, st.session_state.messages)
            response = result["response"]
            tool_calls = result.get("tool_calls", 0)

            st.markdown(f"""
            <div class="chat-box-assistant">
            {response}
            </div>
            """, unsafe_allow_html=True)

            # Show tool usage indicator
            if tool_calls > 0:
                st.caption(f"✨ Used {tool_calls} tool(s) to answer this question")

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}\n\nPlease try rephrasing your question."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("⚕️ CVS HealthHub AI - Powered by LangGraph, ChromaDB, and OpenRouter | For informational purposes only")
