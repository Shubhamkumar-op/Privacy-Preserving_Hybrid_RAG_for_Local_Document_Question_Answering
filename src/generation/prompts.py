SYSTEM_PROMPT = """You are a private local document assistant.
Answer ONLY from the supplied document context.
If the context does not contain enough information, say that the answer was not found in the documents.
Do not invent facts or use outside knowledge.
Always cite the PDF name and page number for factual claims."""


def build_prompt(question: str, contexts) -> str:
    context = "\n\n".join(
        f"[{chunk.pdf_name}, page {chunk.page_number}]\n{chunk.text}" for chunk, _ in contexts
    )
    return f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"
