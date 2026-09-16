```python
import os
import io
import re
import hashlib

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    api_key = None

    # Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Local environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = clean_text(text)

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(texts, model):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# =========================================================
# CREATE FAISS DATABASE
# =========================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# SEARCH FAISS
# =========================================================

def search_faiss(
    question,
    model,
    index,
    chunks,
    top_k=TOP_K
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score)
        })

    return results


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    context = ""

    for i, result in enumerate(results, start=1):

        context += f"""
SOURCE {i}
PAGE: {result['page']}

{result['text']}

------------------------------
"""

    return context


# =========================================================
# GROQ RESPONSE
# =========================================================

def ask_groq(client, question, context):

    system_prompt = """
You are a PDF document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not present in the document,
  say that the information was not found in the
  uploaded document.
- Give clear and useful answers.
- Mention the relevant page number when possible.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

========================================

QUESTION:

{question}

========================================

Answer using only the document context.
"""

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_tokens=1200
    )

    return response.choices[0].message.content


# =========================================================
# SESSION STATE
# =========================================================

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "file_hash" not in st.session_state:
    st.session_state.file_hash = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# TITLE
# =========================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        "**Vector Database:** `FAISS`"
    )

    st.write(
        f"**Top-K:** `{TOP_K}`"
    )

    st.write(
        f"**Chunk Size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Chunk Overlap:** `{CHUNK_OVERLAP}`"
    )


# =========================================================
# GROQ API CHECK
# =========================================================

groq_client = get_groq_client()

if groq_client is None:

    st.warning(
        "⚠️ GROQ_API_KEY is not configured."
    )


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    # Process only if new PDF
    if current_hash != st.session_state.file_hash:

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                # -------------------------------
                # 1. Extract
                # -------------------------------

                pages = extract_pdf_text(
                    pdf_bytes
                )

                if not pages:

                    st.error(
                        "No readable text found in this PDF."
                    )

                    st.stop()

                # -------------------------------
                # 2. Chunk
                # -------------------------------

                chunks = create_chunks(
                    pages
                )

                if not chunks:

                    st.error(
                        "Could not create chunks."
                    )

                    st.stop()

                # -------------------------------
                # 3. Embedding model
                # -------------------------------

                model = load_embedding_model()

                # -------------------------------
                # 4. Embeddings
                # -------------------------------

                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = create_embeddings(
                    texts,
                    model
                )

                # -------------------------------
                # 5. FAISS
                # -------------------------------

                index = create_faiss_index(
                    embeddings
                )

                # -------------------------------
                # Save
                # -------------------------------

                st.session_state.faiss_index = index

                st.session_state.chunks = chunks

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.file_hash = (
                    current_hash
                )

                st.session_state.messages = []

                st.success(
                    f"✅ PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    "❌ Error while processing PDF."
                )

                st.exception(e)


# =========================================================
# DOCUMENT STATUS
# =========================================================

if st.session_state.faiss_index is not None:

    st.success(
        f"📄 Document: "
        f"{st.session_state.document_name}"
    )

    st.info(
        f"🔹 Total chunks: "
        f"{len(st.session_state.chunks)}"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

if st.session_state.faiss_index is not None:

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # -------------------------------
        # USER MESSAGE
        # -------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        # -------------------------------
        # CHECK API
        # -------------------------------

        if groq_client is None:

            with st.chat_message("assistant"):

                st.error(
                    "GROQ_API_KEY is missing. "
                    "Add it to Streamlit Secrets."
                )

        else:

            with st.chat_message("assistant"):

                try:

                    # -------------------------------
                    # Load model
                    # -------------------------------

                    model = load_embedding_model()

                    # -------------------------------
                    # Retrieval
                    # -------------------------------

                    with st.spinner(
                        "🔎 Searching document..."
                    ):

                        results = search_faiss(
                            question,
                            model,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            TOP_K
                        )

                    # -------------------------------
                    # Context
                    # -------------------------------

                    context = build_context(
                        results
                    )

                    # -------------------------------
                    # Groq
                    # -------------------------------

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        answer = ask_groq(
                            groq_client,
                            question,
                            context
                        )

                    # -------------------------------
                    # Answer
                    # -------------------------------

                    st.markdown(answer)

                    # -------------------------------
                    # Sources
                    # -------------------------------

                    if results:

                        with st.expander(
                            "📖 View Retrieved Sources"
                        ):

                            for i, result in enumerate(
                                results,
                                start=1
                            ):

                                st.markdown(
                                    f"""
**Source {i} — Page {result['page']}**

Similarity: `{result['score']:.3f}`

{result['text']}
"""
                                )

                    # -------------------------------
                    # Save answer
                    # -------------------------------

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        "❌ Error while generating response."
                    )

                    st.exception(e)

else:

    st.info(
        "👆 Upload a PDF to start."
    )
```
