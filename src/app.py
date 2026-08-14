import os
import tempfile

import streamlit as st

from src.rag import RAGPipeline
from src.translation.translator import translate_to_hindi


st.set_page_config(page_title="Privacy-Preserving Hybrid RAG", page_icon="🔒", layout="wide")

st.title("🔒 Privacy-Preserving Hybrid RAG")
st.caption("Local PDF question answering using BM25 + FAISS + Ollama")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    paths = []
    for uploaded in uploaded_files:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp.write(uploaded.getbuffer())
        temp.close()
        paths.append(temp.name)

    if st.button("Process documents", type="primary"):
        with st.spinner("Extracting, chunking and indexing documents..."):
            try:
                st.session_state.pipeline.ingest_many(paths)
                st.session_state.documents_ready = True
                st.success(f"Processed {len(uploaded_files)} PDF(s).")
            except Exception as exc:
                st.error(str(exc))

question = st.text_input("Ask a question about your documents")

if question and st.session_state.get("documents_ready", False):
    with st.spinner("Retrieving evidence and generating a local answer..."):
        try:
            answer, contexts = st.session_state.pipeline.ask(question)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved evidence")
            for chunk, score in contexts:
                st.markdown(f"**{chunk.pdf_name} — Page {chunk.page_number}** · score: `{score:.4f}`")
                st.write(chunk.text)

            if st.button("Translate answer to Hindi"):
                with st.spinner("Translating locally..."):
                    st.subheader("Hindi translation")
                    st.write(translate_to_hindi(answer))
        except Exception as exc:
            st.error(str(exc))
elif question:
    st.info("Process at least one PDF before asking a question.")
