# app_streamlit_full.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from groq_remote_llm import GroqRemoteLLM
from langchain.chains import RetrievalQA

st.set_page_config(page_title="PDF QA (Answer + Sources)", layout="wide")
st.title("PDF QA — Answer + Sources")

INDEX_DIR = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_resources():
    emb = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    # if you saved with pickle and you trust it:
    vectorstore = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = GroqRemoteLLM()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    return qa, retriever

qa, retriever = load_resources()

# UI: left column = input, right column = logs / history
col1, col2 = st.columns([3,1])

with col1:
    q = st.text_input("Ask a question about the document")
    if st.button("Ask") and q:
        with st.spinner("Thinking..."):
            try:
                res = qa.invoke({"query": q})
                # robust extraction
                answer = res.get("output_text") or res.get("result") or next((v for v in res.values() if isinstance(v, str)), "")
                st.subheader("Answer")
                st.write(answer)
                st.markdown("---")
                st.subheader("Retrieved source chunks")
                docs = res.get("source_documents") or retriever.get_relevant_documents(q)
                for i, d in enumerate(docs, 1):
                    src = getattr(d, "metadata", {}).get("source", "unknown")
                    st.markdown(f"**Source {i} — {src}**")
                    st.write(d.page_content[:1000])
            except Exception as e:
                st.error(f"Error running QA: {e}")

with col2:
    st.markdown("### Quick Options")
    k = st.number_input("Retriever k", min_value=1, max_value=10, value=2, step=1)
    st.write("Change retriever k then Reload app to take effect.")
    st.markdown("---")
    st.markdown("### Notes")
    st.write("Model:", os.getenv('GROQ_MODEL'))
    st.write("Index:", INDEX_DIR)
    st.write("Use small k (1-3) for factual lookups; use map_reduce for longer answers.")