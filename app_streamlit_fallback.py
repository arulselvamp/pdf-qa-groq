# app_streamlit_fallback.py — safe fallback to restore UI immediately
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="PDF Q&A — Fallback", layout="centered")
st.title("PDF Q&A — Fallback mode")

st.info("This is fallback mode. The full app was temporarily disabled to avoid startup crashes. "
        "Click 'Index & Query' after uploading a PDF to safely run the pipeline.")

uploaded_file = st.file_uploader("Upload PDF (safe mode)", type=["pdf"])
if uploaded_file:
    st.write("File ready:", uploaded_file.name)
    if st.button("Index & Query (safe)"):
        try:
            # Lazy import heavy libs only when user explicitly asks
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            from langchain_community.vectorstores import FAISS
            import os, requests, json

            # Save uploaded file
            pdf_path = "uploaded_safe.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            st.write("Splitting PDF into chunks (safe)...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            docs = splitter.split_documents(pages)
            st.write(f"Created {len(docs)} chunks.")

            st.write("Creating embeddings & FAISS (may take a moment)...")
            emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vs = FAISS.from_documents(docs, emb)
            st.session_state["vs_safe"] = vs
            st.success("Index created successfully in session (safe).")

            q = st.text_input("Ask a question about the uploaded PDF", key="q_safe")
            if q and st.button("Ask now"):
                retriever = st.session_state["vs_safe"].as_retriever(search_kwargs={"k": 3})
                docs_r = retriever.get_relevant_documents(q)
                context = "\n\n".join([d.page_content for d in docs_r])[:3000]
                # Simple Groq call (check env)
                API_URL = os.getenv("GROQ_API_URL")
                API_KEY = os.getenv("GROQ_API_KEY")
                if not API_URL or not API_KEY:
                    st.error("Missing GROQ_API_URL or GROQ_API_KEY in env/secrets.")
                else:
                    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                    payload = {"model": os.getenv("GROQ_MODEL"), "messages":[{"role":"user","content": f"Context:{context}\n\nQ:{q}"}], "max_tokens": 200}
                    resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                    ans = resp.json()
                    out = ans.get("choices",[{}])[0].get("message",{}).get("content","(no text)")
                    st.subheader("Answer (safe mode)")
                    st.write(out)
        except Exception as e:
            st.error("Safe-mode operation failed. Check app logs for details.")
            st.exception(e)
else:
    st.write("Upload a PDF to start (safe mode).")
