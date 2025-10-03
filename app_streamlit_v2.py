# app_streamlit_v2.py
"""
Improved Streamlit PDF Q&A (Groq LLaMA) — v2
Features:
 - Structured JSON answer (conclusion, supporting_snippet, source)
 - Supporting quote extraction
 - Better error handling and caching
 - Session history + CSV download
 - Sidebar controls for chunking, retriever k, max_tokens, temperature
"""
from dotenv import load_dotenv
load_dotenv()  # local dev only; Streamlit Cloud uses Secrets

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

# ------------------ GroqRemoteLLM (robust) ------------------
class GroqRemoteLLM:
    def __init__(self, api_url=None, api_key=None, model=None, timeout=60, temperature=0.0):
        self.api_url = api_url or os.getenv("GROQ_API_URL")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout
        self.temperature = float(temperature)

    def _call(self, prompt: str, max_tokens: int = 256) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("GROQ_API_URL or GROQ_API_KEY not set (Streamlit Secrets or .env).")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer only from the provided context."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(self.temperature),
        }
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        out = resp.json()
        # robust extraction of assistant text
        if isinstance(out, dict):
            choices = out.get("choices") or out.get("responses") or []
            if isinstance(choices, list) and len(choices) > 0:
                c0 = choices[0]
                # common shape: { "message": {"content":"..."} }
                if isinstance(c0, dict):
                    if isinstance(c0.get("message"), dict):
                        return c0["message"].get("content", "")
                    if "text" in c0:
                        return c0.get("text", "")
        # fallback: stringify
        return json.dumps(out)

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        return self._call(prompt, max_tokens=max_tokens)


# ------------------ Utilities ------------------
def file_hash(bytes_data: bytes) -> str:
    return hashlib.md5(bytes_data).hexdigest()


@st.cache_resource
def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)


@st.cache_resource
def build_faiss(docs: List, embed_model_name: str = "all-MiniLM-L6-v2"):
    emb = get_embeddings(embed_model_name)
    vs = FAISS.from_documents(docs, emb)
    return vs


def split_pdf_to_docs(pdf_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def parse_json_safe(raw: str):
    # Attempt to extract JSON object from raw text
    try:
        return json.loads(raw)
    except Exception:
        # try to find a JSON object substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                return None
        return None


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Q&A v2 (Groq LLaMA)", layout="wide")
st.title("📄 PDF Q&A — v2 (Structured + Supporting Quote)")

# Sidebar controls
st.sidebar.header("Settings")
CHUNK_SIZE = st.sidebar.number_input("Chunk size (chars)", 300, 2000, 1000, step=100)
CHUNK_OVERLAP = st.sidebar.number_input("Chunk overlap (chars)", 0, 400, 100, step=20)
RETRIEVER_K = st.sidebar.number_input("Retriever k (top-k)", 1, 10, 3, step=1)
MAX_TOKENS = st.sidebar.number_input("Max tokens (LLM)", 64, 1024, 256, step=32)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
EMBED_MODEL = st.sidebar.selectbox("Embedding model", options=["all-MiniLM-L6-v2"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Put GROQ_API_KEY and GROQ_API_URL in Streamlit Secrets (Manage app → Secrets).")

# Upload & question
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

# initialize history
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# main flow
if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        key = file_hash(file_bytes)
        pdf_path = f"uploaded_{key}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        cache_key = f"docs-{key}-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
        if cache_key in st.session_state:
            docs = st.session_state[cache_key]
        else:
            with st.spinner("Splitting PDF into chunks..."):
                docs = split_pdf_to_docs(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)
                st.session_state[cache_key] = docs

        vs_key = f"vs-{key}-{EMBED_MODEL}-{len(docs)}"
        if vs_key in st.session_state:
            vectorstore = st.session_state[vs_key]
        else:
            with st.spinner("Building embeddings and FAISS index (cached)..."):
                vectorstore = build_faiss(docs, embed_model_name=EMBED_MODEL)
                st.session_state[vs_key] = vectorstore

        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        st.success(f"PDF indexed: {len(docs)} chunks available.")
    except Exception as e:
        st.error("Failed to process uploaded PDF. See details below.")
        st.exception(e)
        retriever = None
        docs = None

    # Answering
    if query and retriever:
        llm = GroqRemoteLLM(temperature=TEMPERATURE)
        # build context with chunk separators and indices
        with st.spinner("Retrieving top chunks..."):
            retrieved = retriever.get_relevant_documents(query)
            # include chunk index labels to help model cite source
            context_parts = []
            for i, d in enumerate(retrieved):
                label = f"CHUNK {i+1}\n{d.page_content}"
                context_parts.append(label)
            context = "\n\n----\n\n".join(context_parts)

        # prompt asks for JSON with supporting snippet and source
        prompt_template = """
You are given the following document context (top retrieved chunks):

{context}

Question: {question}

Task:
- Extract a concise factual answer (one or two sentences) to the question from the context.
- Also return the exact supporting sentence from the context that justifies the answer under the key "supporting_snippet".
- Include "source" with the chunk label (e.g., "CHUNK 2") where the supporting sentence came from.
- Return the output strictly as a JSON object with keys: "answer", "supporting_snippet", "source".
- If the answer is not present in the context, set "answer" to "Not stated in the document" and other keys to null.

Important: Only return the JSON (no extra commentary).
"""
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        prompt = QA_PROMPT.format(context=context, question=query)

        # call LLM
        try:
            raw = llm(prompt, max_tokens=MAX_TOKENS)
        except Exception as e:
            st.error("LLM call failed. See details below.")
            st.exception(e)
            raw = None

        parsed = None
        if raw:
            parsed = parse_json_safe(raw)

        # fallback: if parsing failed, ask model again in a simpler form
        if not parsed:
            with st.spinner("Parsing JSON failed; trying a simpler extraction prompt..."):
                try:
                    simple_prompt = (
                        "From the context below, answer the question in one sentence, then on the next line "
                        "write 'Support:' and paste the exact supporting sentence from the context (or 'Not stated').\n\n"
                        f"Context:\n{context}\n\nQuestion: {query}\n"
                    )
                    raw2 = llm(simple_prompt, max_tokens=MAX_TOKENS)
                except Exception as e:
                    st.error("Fallback LLM call failed.")
                    st.exception(e)
                    raw2 = None

                # try to split raw2 into answer + support
                if raw2:
                    # naive split on "Support:"
                    if "Support:" in raw2:
                        a, s = raw2.split("Support:", 1)
                        parsed = {
                            "answer": a.strip(),
                            "supporting_snippet": s.strip(),
                            "source": None
                        }
                    else:
                        parsed = {
                            "answer": raw2.strip(),
                            "supporting_snippet": None,
                            "source": None
                        }

        # Show result to user
        st.subheader("Answer (structured)")
        if parsed:
            st.json(parsed)
            # show as readable
            st.markdown("**Answer:**")
            st.write(parsed.get("answer"))
            if parsed.get("supporting_snippet"):
                with st.expander("Supporting sentence"):
                    st.write(parsed.get("supporting_snippet"))
                    if parsed.get("source"):
                        st.caption(f"Source: {parsed.get('source')}")
        else:
            st.error("Model did not return a parseable answer.")
            if raw:
                st.code(raw)

        # grounding check (token overlap)
        try:
            toks = [t for t in parsed.get("answer", "").split() if len(t) > 2][:40] if parsed else []
            combined = " ".join([d.page_content for d in retrieved]).lower() if retrieved else ""
            found = [t for t in toks if t.lower().strip(".,") in combined]
            grounding_pct = round(100.0 * (len(found) / max(1, len(toks))), 1)
            st.info(f"Grounding: {grounding_pct}% of important answer tokens found in retrieved chunks ({len(found)}/{len(toks)})")
        except Exception:
            pass

        # Save to session history
        st.session_state["qa_history"].append({
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "question": query,
            "answer": parsed.get("answer") if parsed else (raw or ""),
            "supporting_snippet": parsed.get("supporting_snippet") if parsed else None,
            "source": parsed.get("source") if parsed else None,
        })

# History UI + Download
st.markdown("---")
st.subheader("Session Q/A history")
if st.session_state["qa_history"]:
    df = pd.DataFrame(st.session_state["qa_history"])
    st.dataframe(df[["timestamp", "question", "answer", "source"]].sort_values(by="timestamp", ascending=False))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Q/A history (CSV)", data=csv_bytes, file_name="qa_history_v2.csv", mime="text/csv")
else:
    st.write("No Q/A items in this session yet.")

# Debug (sidebar toggle)
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("GROQ_API_URL set:", bool(os.getenv("GROQ_API_URL")))
    st.sidebar.write("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("Indexed docs cache keys:", [k for k in st.session_state.keys() if str(k).startswith("vs-") or str(k).startswith("docs-")])