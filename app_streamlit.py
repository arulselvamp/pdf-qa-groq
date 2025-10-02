# app_streamlit.py (simplified)
# Top of app_streamlit.py
from dotenv import load_dotenv
import os
load_dotenv()   # reads .env into os.environ

# Optional debug
print("Loaded GROQ_API_URL:", os.getenv("GROQ_API_URL"))


import streamlit as st
from groq_remote_llm import GroqRemoteLLM
from langchain.chains import RetrievalQA
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
#from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


st.title("PDF QA (Groq Llama)")

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    with open("tmp.pdf", "wb") as f:
        f.write(uploaded.getbuffer())
    loader = PyPDFLoader("tmp.pdf")
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    llm = GroqRemoteLLM()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    q = st.text_input("Ask a question")
    if q:
        with st.spinner("Thinking..."):
            ans = qa.run(q)
        st.write(ans)
