"""
CVS HealthHub AI - Streamlit Frontend
Interactive chat interface for healthcare agent
"""
import streamlit as st
import sys
import os
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.healthhub_agent import chat, vector_store, analyze_prescription_image

# Page configuration
st.set_page_config(
    page_title="CVS HealthHub AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme Compatible
st.markdown("""
<style>
    /* Root Variables for Theme Support */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #161b22;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --border-color: #30363d;
        --accent-red: #da3633;
        --accent-blue: #58a6ff;
    }

    /* Main Background and Text */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #ff4444;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 1px 1px 3px rgba(255,0,0,0.2);
    }

    .sub-header {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid;
    }

    /* User Message */
    .stChatMessage[data-testid="chatMessage"] {
        background-color: var(--bg-secondary) !important;
        border-left-color: var(--accent-blue) !important;
    }

    .stChatMessage[data-testid="chatMessage"] p,
    .stChatMessage[data-testid="chatMessage"] span {
        color: var(--text-primary) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ff4444;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--text-primary) !important;
    }

    /* Button Styling */
    .stButton button {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }

    .stButton button:hover {
        border-color: #ff4444;
        color: #ff4444;
        background-color: rgba(255,68,68,0.1);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(255,68,68,0.2);
    }

    /* Primary Button (Analyze) */
    button[kind="primary"] {
        background-color: #ff4444 !important;
        color: white !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background-color: #ff5555 !important;
    }

    /* Text Input Styling */
    .stTextInput input,
    .stTextArea textarea {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }

    /* Select and Dropdown */
    select {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }

    .stChatInputContainer textarea {
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #ff4444 !important;
        box-shadow: 0 4px 15px rgba(255,68,68,0.2) !important;
    }

    /* Status Indicators */
    .stAlert {
        border-radius: 8px;
        border: none;
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    /* Headers and Text */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    p, span, label {
        color: var(--text-primary) !important;
    }

    .stInfo {
        background-color: rgba(88, 166, 255, 0.1) !important;
        border-color: var(--accent-blue) !important;
    }

    .stSuccess {
        background-color: rgba(26, 188, 156, 0.1) !important;
        border-color: #1abc9c !important;
    }

    .stWarning {
        background-color: rgba(241, 196, 15, 0.1) !important;
        border-color: #f1c40f !important;
    }

    .stError {
        background-color: rgba(255, 68, 68, 0.1) !important;
        border-color: #ff4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏥 CVS HealthHub AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Knowledge Assistant for Digital Workplace</div>', unsafe_allow_html=True)

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
    - 📸 **NEW!** Prescription image analysis

    *This is informational only - always consult healthcare providers for medical decisions.*
    """)

    st.header("Analyze Prescription Image")
    st.write("Upload a prescription or medication photo")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload prescription, medication bottle, or pharmacy label"
    )

    if uploaded_file is not None:
        # Display image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("🤖 Analyzing with NVIDIA AI..."):
                try:
                    # Convert to base64
                    image_bytes = uploaded_file.read()
                    image_b64 = base64.b64encode(image_bytes).decode()

                    # Analyze
                    result = analyze_prescription_image(image_b64, "base64")

                    if result["success"]:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📸 **Image Analysis Results:**\n\n{result['analysis']}"
                        })
                        st.success(" Analysis complete! Check chat below.")
                        st.rerun()
                    else:
                        st.error(f"{result.get('error', 'Analysis failed')}")
                except Exception as e:
                    st.error(f" Error: {str(e)}")

    st.divider()

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
        st.info("🤖 NVIDIA Multimodal AI: Active")
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
- 💊 Medication information and side effects
- ⚕️ Drug interaction checks
- 💉 Vaccine availability and scheduling
- 🏪 CVS pharmacy services and hours
- 💳 Insurance coverage questions
- 📸  Prescription image analysis (upload in sidebar)

**Try uploading a prescription image** in the sidebar for AI-powered analysis using NVIDIA's multimodal model!

What would you like to know?"""
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle selected query from sidebar
if "selected_query" in st.session_state:
    user_input = st.session_state.selected_query
    del st.session_state.selected_query
else:
    user_input = st.chat_input("Ask about medications, vaccines, services, or insurance...")

# Process user input - only if there's actual new input
if user_input and user_input.strip():
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response (pass history BEFORE adding current message)
    with st.spinner("🔍 Searching knowledge base and analyzing..."):
        try:
            result = chat(user_input, st.session_state.messages)
            response = result["response"]
            tool_calls = result.get("tool_calls", 0)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

                # Show tool usage indicator
                if tool_calls > 0:
                    st.caption(f"✨ Used {tool_calls} tool(s) to answer this question")

            # Add both messages to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}\n\nPlease try rephrasing your question."
            st.error(error_msg)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("⚕️ CVS HealthHub AI - Powered by LangGraph, ChromaDB, and OpenRouter | For informational purposes only")
