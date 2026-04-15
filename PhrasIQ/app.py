import streamlit as st
import os
from part1 import build_rag_system
from part2 import build_mitigated_rag_system

# Page Configuration
st.set_page_config(page_title="PhrasIQ Financial RAG", page_icon="📈", layout="wide")

# Sidebar for Version Selection (Demo Optimization)
st.sidebar.title("Demo Controls")
version = st.sidebar.radio(
    "Select System Architecture:",
    ("Standard RAG (Part 1)", "Mitigated RAG (Part 2)"),
    help="Part 2 includes the 'Auditor' layer to prevent hallucinations on edge cases."
)

st.title("📈 PhrasIQ Financial Analyst AI (GPT-5)")

if version == "Standard RAG (Part 1)":
    st.info("Currently running **Standard RAG** with **GPT-5-mini**.")
    rag_fn = build_rag_system
else:
    st.success("Currently running **Mitigated RAG (v2)** with **GPT-5-mini**.")
    rag_fn = build_mitigated_rag_system

# Initialize the selected RAG system
@st.cache_resource
def get_rag_chain(ver_name):
    # Returns the selected system (hardcoded to GPT-5 internally)
    return rag_fn()

try:
    with st.spinner("Initializing AI Engine..."):
        chain = get_rag_chain(version)
    st.success("AI Engine (GPT-5) Ready!")
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Query the Data")
    user_query = st.text_input("Ask a question about Q3 performance:", 
                                value="Which business unit had the largest unfavorable variance in Q3, and what factors does the commentary attribute this to?")
    
    if st.button("Analyze"):
        if user_query:
            with st.spinner("Analyzing data and generating response..."):
                response = chain.invoke(user_query)
                
                st.markdown("### Answer")
                st.write(response["answer"])
                
                with col2:
                    st.subheader("Source Grounding")
                    st.info("The answer is derived from the following sources:")
                    seen_sources = set()
                    for i, doc in enumerate(response["context"]):
                        source_id = doc.metadata.get("doc_id") or doc.metadata.get("source")
                        if source_id not in seen_sources:
                            with st.expander(f"📍 {source_id}"):
                                st.text_area(f"Context Snippet {i+1}", doc.page_content, height=150)
                            seen_sources.add(source_id)
        else:
            st.warning("Please enter a question.")

with col2:
    if not user_query:
        st.subheader("Available Data")
        st.markdown("""
        - **bu_financials_q3.csv**: Revenue, EBITDA, and Headcount metrics.
        - **commentary_excerpts.txt**: Internal variance commentary and QBR transcripts.
        """)

# Custom Styling
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
