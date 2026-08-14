import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_client_cls" in .* has conflict with protected namespace "model_".*',
    category=UserWarning
)

import os
import fitz
import faiss
import pickle
import textwrap
import streamlit as st

from sentence_transformers import SentenceTransformer
from autogen import ConversableAgent


LLM_CONFIG = {
    "model": "mistral:latest",
    "base_url": "http://localhost:11434",
    "api_type": "ollama"
}

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

STORAGE_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "pdf_data"
)

os.makedirs(STORAGE_DIR, exist_ok=True)


class PDFAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_by_page(self):
        try:
            doc = fitz.open(self.pdf_path)
            pages = [
                (i + 1, page.get_text())
                for i, page in enumerate(doc)
            ]
            doc.close()
            return pages
        except Exception as e:
            st.error(f"PDF error: {e}")
            return []


class EmbeddingAgent:
    def __init__(self, pdf_name):
        self.pdf_name = pdf_name
        self.emb_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def chunk_text(self, pages, chunk_size=200):
        chunks = []

        for page_num, text in pages:
            wrapped = textwrap.wrap(
                text,
                width=chunk_size
            )

            for chunk in wrapped:
                if chunk.strip():
                    chunks.append(
                        (page_num, chunk)
                    )

        return chunks

    def build_index(self, chunks):
        texts = [chunk for _, chunk in chunks]

        embeddings = self.emb_model.encode(
            texts,
            convert_to_numpy=True
        )

        index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        index.add(embeddings)

        return index

    def store(self, index, chunks):
        base = os.path.join(
            STORAGE_DIR,
            self.pdf_name
        )

        faiss.write_index(
            index,
            f"{base}.index"
        )

        with open(
            f"{base}.pkl",
            "wb"
        ) as f:
            pickle.dump(chunks, f)


def store_pdf_embedding(pdf_path):
    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    base = os.path.join(
        STORAGE_DIR,
        pdf_name
    )

    if (
        os.path.exists(f"{base}.index")
        and os.path.exists(f"{base}.pkl")
    ):
        return False

    pages = PDFAgent(
        pdf_path
    ).extract_text_by_page()

    if not pages:
        return False

    agent = EmbeddingAgent(pdf_name)

    chunks = agent.chunk_text(pages)

    index = agent.build_index(chunks)

    agent.store(index, chunks)

    return True


class QueryAgent:
    def __init__(self):
        self.emb_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def search_all(self, query, top_k=5):
        query_vec = self.emb_model.encode(
            [query],
            convert_to_numpy=True
        )

        results = []

        for file in os.listdir(STORAGE_DIR):

            if not file.endswith(".index"):
                continue

            pdf_name = file[:-6]

            index_path = os.path.join(
                STORAGE_DIR,
                file
            )

            pickle_path = os.path.join(
                STORAGE_DIR,
                f"{pdf_name}.pkl"
            )

            try:
                index = faiss.read_index(
                    index_path
                )

                with open(
                    pickle_path,
                    "rb"
                ) as f:
                    chunks = pickle.load(f)

                distances, indices = index.search(
                    query_vec,
                    top_k
                )

                for distance, idx in zip(
                    distances[0],
                    indices[0]
                ):
                    if 0 <= idx < len(chunks):
                        page, chunk = chunks[idx]

                        results.append({
                            "pdf_name": pdf_name,
                            "page": page,
                            "text": chunk,
                            "distance": float(distance)
                        })

            except Exception as e:
                st.warning(
                    f"Error reading {pdf_name}: {e}"
                )

        results.sort(
            key=lambda x: x["distance"]
        )

        return results[:top_k]


def generate_final_answer(query, contexts):
    agent = ConversableAgent(
        name="AnswerAgent",
        system_message=(
            "Answer using the provided PDF evidence. "
            "Mention the PDF name and page number. "
            "Do not invent information."
        ),
        llm_config=LLM_CONFIG
    )

    context = "\n\n".join(
        f"PDF: {item['pdf_name']}\n"
        f"Page: {item['page']}\n"
        f"Content: {item['text']}"
        for item in contexts
    )

    prompt = f"""
Retrieved PDF evidence:

{context}

Question:
{query}

Answer the question using the evidence above.
Mention relevant PDF names and page numbers.
"""

    response = agent.generate_reply(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.get(
        "content",
        "No answer generated."
    )


def main():
    st.set_page_config(
        page_title="Privacy-Preserving Hybrid RAG",
        page_icon="🔐",
        layout="wide"
    )

    st.title(
        "🔐 Privacy-Preserving Hybrid RAG"
    )

    st.write(
        "Local document question answering using "
        "FAISS and Ollama."
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:

            pdf_path = os.path.join(
                STORAGE_DIR,
                uploaded_file.name
            )

            if not os.path.exists(pdf_path):

                with open(
                    pdf_path,
                    "wb"
                ) as f:
                    f.write(
                        uploaded_file.getbuffer()
                    )

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):
                    store_pdf_embedding(
                        pdf_path
                    )

                st.sidebar.success(
                    f"✓ {uploaded_file.name}"
                )

            else:
                st.sidebar.info(
                    f"{uploaded_file.name} already loaded"
                )

    pdf_files = [
        f
        for f in os.listdir(STORAGE_DIR)
        if f.endswith(".pdf")
    ]

    st.sidebar.subheader("Loaded PDFs")

    for pdf in pdf_files:
        st.sidebar.write(f"📄 {pdf}")

    if "agent" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.agent = QueryAgent()

    query = st.text_input(
        "Ask a question"
    )

    if query:

        with st.spinner("Searching..."):
            contexts = st.session_state.agent.search_all(
                query,
                top_k=5
            )

        if not contexts:
            st.warning(
                "No relevant information found."
            )
            return

        st.subheader(
            "📄 Retrieved Evidence"
        )

        for item in contexts:

            st.markdown(
                f"""
                **📘 {item['pdf_name']} — Page {item['page']}**

                {item['text']}
                """
            )

        with st.spinner(
            "Generating answer..."
        ):
            answer = generate_final_answer(
                query,
                contexts
            )

        st.subheader(
            "💡 Answer"
        )

        st.markdown(answer)


if __name__ == "__main__":
    main()