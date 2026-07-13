import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Document Q&A (RAG)")
st.title("Document Question Answering System")
st.caption("Retrieval-Augmented Generation over your own documents (e.g. Operating System Concepts — Galvin)")

if not os.environ.get("GROQ_API_KEY"):
    st.info(
        "No GROQ_API_KEY found in your environment. The app will still run, but "
        "will show the top-matching passage instead of a fully generated answer. "
        "Get a free key at console.groq.com and add it to a .env file to enable "
        "full answer generation."
    )


@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline(index_dir="index")
    if not Path("index/index.faiss").exists():
        return None
    pipeline.load_index()
    return pipeline


pipeline = load_pipeline()

if pipeline is None:
    st.warning(
        "No index found yet. Add a PDF to the `data/` folder, then run "
        "`python build_index.py` in your terminal, and refresh this page.",
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.write(f"- {s['source']}, page {s['page']} (score: {s['score']})")

question = st.chat_input("Ask a question about your document...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant sections and generating answer..."):
            result = pipeline.ask(question)
        st.write(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.write(f"- {s['source']}, page {s['page']} (score: {s['score']})")

    st.session_state.history.append(result)
