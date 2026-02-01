"""
Streamlit UI for Company AI Assistant.

A clean, professional chat interface for querying company documents.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import query
from src.config import Config

# Page config
st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Source citations */
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    /* Stats */
    .stats-box {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def handle_user_question(question: str):
    # 1. Add user message immediately
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. Mark question as pending (assistant not answered yet)
    st.session_state.pending_question = question

    # 3. Rerun so UI updates immediately
    st.rerun()


# Sidebar
with st.sidebar:
    st.markdown("# ℹ️ About")
    st.info(
        """This AI assistant can answer questions about:\n
- Remote work policies
- Leave and time off
- Expense reporting
- Travel guidelines
- Security procedures
- And more..."""
    )

    st.divider()

    st.markdown("## Stats")
    st.metric("Total Queries", st.session_state.total_queries)
    st.metric("Total Tokens Used", st.session_state.total_tokens)

    st.divider()

    # Configuration
    with st.expander("⚙️ Settings"):
        st.write(f"**Model:** {Config.OPENAI_MODEL}")
        st.write(f"**Environment:** {Config.ENVIRONMENT.value}")
        st.write(f"**Top-K Results:** {Config.RETRIEVAL_TOP_K}")
        st.write(f"**Min Score:** {Config.RETRIEVAL_MIN_SCORE}")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: #666; font-size: 14px;'>
        Built with Streamlit • Powered by OpenAI • Vector search with pgvector
    </div>
    """,
        unsafe_allow_html=True,
    )

# Centered layout (ChatGPT-style)
left, center, right = st.columns([1, 3, 1])

with center:
    # Main chat interface
    st.title("🧠 AI Company Policy Assistant")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f"<strong>{source['filename']}</strong><br>"
                            f"Chunks used: {', '.join(map(str, source['chunks_used']))}<br>"
                            f"Relevance score: {source['max_score']:.3f}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Show stats if available
            if "stats" in message:
                stats = message["stats"]
                st.markdown(
                    f'<div class="stats-box">'
                    f"📊 {stats['chunks']} chunks retrieved | "
                    f"💬 {stats['tokens']} tokens used | "
                    f"⏱️ {stats['time']:.2f}s"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Chat input
    user_question = st.chat_input("Ask a question about company policies...")

    if user_question:
        handle_user_question(user_question)

    # Example questions (show when chat is empty)
    if not st.session_state.messages:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📅 What is the remote work policy?"):
                handle_user_question("What is the remote work policy?")

            if st.button("💰 When are expense reports due?"):
                handle_user_question("When are expense reports due?")

        with col2:
            if st.button("🏖️ How many vacation days do I get?"):
                handle_user_question("How many vacation days do I get?")

            if st.button("🔒 What are the security requirements?"):
                handle_user_question("What are the security requirements?")

        # If there is a pending question, generate the assistant response
    if st.session_state.pending_question:
        with st.spinner("🔍 Retrieving your answer..."):
            import time

            question = st.session_state.pending_question
            start_time = time.time()

            result = query(question)

            elapsed_time = time.time() - start_time

        # Append assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "stats": {
                    "chunks": result.get("chunks_retrieved", 0),
                    "tokens": result.get("tokens_used", 0),
                    "time": elapsed_time,
                },
            }
        )

        # Update stats
        st.session_state.total_queries += 1
        st.session_state.total_tokens += result.get("tokens_used", 0)

        # Clear pending question
        st.session_state.pending_question = None

        st.rerun()
