# app_streamlit_demo.py
from dotenv import load_dotenv
load_dotenv()  # for local dev only; Streamlit Cloud uses Secrets

import streamlit as st
import os
import requests
import pandas as pd
import hashlib
import json
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate

# ------------------ Groq wrapper (supports per-call max_tokens) ------------------
class GroqRemoteLLM:
    def __init__(self, api_url=None, api_key=None, model=None, timeout=30, temperature=0.0):
        self.api_url = api_url or os.getenv("GROQ_API_URL")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout
        self.temperature = float(temperature)

    def _call(self, prompt: str, max_tokens: int = 300) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("GROQ_API_URL or GROQ_API_KEY not set in environment/secrets.")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer using only the provided context."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(self.temperature),
        }
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # robust extraction
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                    return first["message"].get("content", "")
                if isinstance(first, dict) and "text" in first:
                    return first.get("text", "")
            # fallback
            for key in ("text", "output_text", "result", "output"):
                if key in data:
                    v = data[key]
                    return v if isinstance(v, str) else json.dumps(v)
        return json.dumps(data)

    def __call__(self, prompt: str, max_tokens: int = 300) -> str:
        return self._call(prompt, max_tokens=max_tokens)


# ------------------ Helpers ------------------
def file_hash(bytes_data: bytes) -> str:
    return hashlib.md5(bytes_data).hexdigest()


@st.cache_resource
def make_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)


@st.cache_resource
def build_vectorstore_from_docs(docs: List, embed_model_name: str = "all-MiniLM-L6-v2"):
    emb = make_embeddings(embed_model_name)
    vs = FAISS.from_documents(docs, emb)
    return vs


def split_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def extract_tokens_for_grounding(answer: str, max_tokens: int = 40):
    # crude tokenizer: words and numbers
    toks = []
    for part in answer.split():
        clean = "".join(ch for ch in part if ch.isalnum() or ch in "._-%")
        if len(clean) >= 2:
            toks.append(clean)
        if len(toks) >= max_tokens:
            break
    return toks


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF QA (Groq LLaMA)", layout="wide")
st.title("📄 PDF Q&A (Groq LLaMA)")

# Sidebar controls
st.sidebar.header("Settings")
CHUNK_SIZE = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=1000, step=100)
CHUNK_OVERLAP = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=400, value=100, step=20)
RETRIEVER_K = st.sidebar.number_input("Retriever k", min_value=1, max_value=10, value=3, step=1)
MAX_TOKENS = st.sidebar.number_input("Max tokens (LLM)", min_value=32, max_value=1024, value=300, step=32)
EMBED_MODEL = st.sidebar.selectbox("Embedding model", options=["all-MiniLM-L6-v2"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Make sure your GROQ_API_KEY and GROQ_API_URL are set in Streamlit Secrets (or .env for local).")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

# Initialize session history
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []  # list of dict rows: {question,answer}

# Process upload and caching vectorstore keyed by file hash
vectorstore = None
docs = None
if uploaded_file:
    file_bytes = uploaded_file.read()
    key = file_hash(file_bytes)
    pdf_path = f"uploaded_{key}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    # cache docs build per file content+settings by key
    cache_key = f"docs-{key}-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
    if cache_key in st.session_state:
        docs = st.session_state[cache_key]
    else:
        with st.spinner("Splitting PDF and creating chunks..."):
            docs = split_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)
            st.session_state[cache_key] = docs

    vs_key = f"vs-{key}-{EMBED_MODEL}-{len(docs)}"
    if vs_key in st.session_state:
        vectorstore = st.session_state[vs_key]
    else:
        with st.spinner("Creating embeddings and vectorstore (may take a moment)..."):
            vectorstore = build_vectorstore_from_docs(docs, embed_model_name=EMBED_MODEL)
            st.session_state[vs_key] = vectorstore

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # Run query
    if query:
        llm = GroqRemoteLLM()
        # Prepare prompt template (context + question)
        template = (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Task: Extract the concise factual answer. Include model names/dataset names and numeric results if present. "
            "If not present, respond exactly: \"Not stated in the document.\""
        )
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Retrieving relevant chunks..."):
            docs_retrieved = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs_retrieved])

        prompt = QA_PROMPT.format(context=context, question=query)

        with st.spinner("Querying Groq LLaMA..."):
            try:
                answer = llm(prompt, max_tokens=MAX_TOKENS)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                answer = f"ERROR: {e}"

        # grounding check
        toks = extract_tokens_for_grounding(answer, max_tokens=30)
        combined = " ".join([d.page_content for d in docs_retrieved]).lower()
        found = [t for t in toks if t.lower() in combined]
        grounding_pct = round(100.0 * (len(found) / max(1, len(toks))), 1)

        # display
        st.subheader("Answer")
        st.write(answer)

        st.info(f"Grounding: {grounding_pct}% of extracted tokens found in retrieved chunks "
                f"({len(found)}/{len(toks)})")

        with st.expander("🔍 Retrieved Chunks (top k)"):
            for i, d in enumerate(docs_retrieved, 1):
                st.markdown(f"**Chunk {i} — source:** {d.metadata.get('source','uploaded')}")
                st.write(d.page_content[:800])

        # Save to history
        st.session_state["qa_history"].append({
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "question": query,
            "answer": answer,
            "grounding_pct": grounding_pct
        })

# History + download
st.markdown("---")
st.subheader("Session Q/A history")
if st.session_state["qa_history"]:
    df = pd.DataFrame(st.session_state["qa_history"])
    st.dataframe(df[["timestamp", "question", "grounding_pct"]].sort_values(by="timestamp", ascending=False))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full Q/A history (CSV)", data=csv_bytes, file_name="qa_history.csv", mime="text/csv")
else:
    st.write("No questions asked in this session yet.")

# Footer / quick debug (only show when user toggles)
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("Env GROQ_API_URL:", bool(os.getenv("GROQ_API_URL")))
    st.sidebar.write("Env GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("Docs count (current):", len(docs) if docs else 0)
    st.sidebar.write("Vectorstore cached:", any(k.startswith("vs-") for k in st.session_state.keys()))
