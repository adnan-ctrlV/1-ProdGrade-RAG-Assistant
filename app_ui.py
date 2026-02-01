"""
Streamlit UI for RAG system.
"""

import streamlit as st
import logging
from src.rag import query

# Configure logging
logging.basicConfig(level="INFO")

# Page config
st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🤖 Company Knowledge Assistant")
st.markdown("Ask questions about company policies and procedures.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This AI assistant can answer questions about:
        - Remote work policies
        - Leave and time off
        - Expense reporting
        - Travel guidelines
        - Security procedures
        - And more...
        """
    )

    st.divider()

    st.header("⚙️ Settings")
    show_sources = st.checkbox("Show source documents", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)

# Main interface
query_input = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is the remote work policy?",
    key="query_input",
)

if st.button("Ask", type="primary", use_container_width=True):
    if not query_input:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base..."):
            result = query(query_input)

        # Display answer
        st.markdown("### 📝 Answer")
        st.markdown(result["answer"])

        # Display sources
        if show_sources and result.get("sources"):
            st.divider()
            st.markdown("### 📚 Sources")

            for source in result["sources"]:
                with st.expander(
                    f"📄 {source['filename']} (score: {source['max_score']:.2f})"
                ):
                    st.markdown(f"**Chunks used:** {source['chunks_used']}")

        # Display metadata
        if show_metadata:
            st.divider()
            st.markdown("### 📊 Metadata")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Chunks Retrieved", result["chunks_retrieved"])

            with col2:
                st.metric("Tokens Used", result["tokens_used"])

            with col3:
                status = "✅ Success" if result["success"] else "❌ Failed"
                st.metric("Status", status)

# Example questions
st.divider()
st.markdown("### 💡 Example Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Remote work policy", use_container_width=True):
        st.session_state.query_input = "What is the remote work policy?"
        st.rerun()

with col2:
    if st.button("Expense reporting", use_container_width=True):
        st.session_state.query_input = "When are expense reports due?"
        st.rerun()

with col3:
    if st.button("Leave policy", use_container_width=True):
        st.session_state.query_input = "How many days of leave do employees get?"
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px;'>
        Built with Streamlit • Powered by OpenAI • Vector search with pgvector
    </div>
    """,
    unsafe_allow_html=True,
)
