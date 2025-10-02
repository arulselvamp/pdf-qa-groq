import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
import os, json

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "sample.pdf"
SAVE_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_MODEL = "all-MiniLM-L6-v2"

if st.button("Rebuild index (use only in dev)"):
    if not os.path.exists(PDF_PATH):
        st.error(f"{PDF_PATH} not found.")
    else:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        st.write(f"Created {len(chunks)} chunks.")
        # save jsonl
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(os.path.join(SAVE_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for doc in chunks:
                rec = {"page_content": doc.page_content, "metadata": getattr(doc, "metadata", {})}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(SAVE_DIR)
        st.success("Rebuilt and saved FAISS index.")
