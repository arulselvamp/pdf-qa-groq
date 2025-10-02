import streamlit as st
import os, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ------------------ Custom LLM Wrapper ------------------
class GroqRemoteLLM:
    def __init__(self, api_url=None, api_key=None, model=None, timeout=30):
        self.api_url = api_url or os.getenv("GROQ_API_URL")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout

    def _call(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        }
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def __call__(self, prompt):
        return self._call(prompt)

# ------------------ Streamlit UI ------------------
st.title("📄 PDF Q&A with Groq LLaMA (70B)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save uploaded file to disk
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load & split
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Build FAISS
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Custom prompt
        template = """
        Context: {context}
        Question: {question}

        Extract the factual answer. If not present, say "Not stated in the document."
        """
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        # Setup LLM
        llm = GroqRemoteLLM()

        # Simple RetrievalQA (manual)
        if query:
            with st.spinner("Querying LLaMA via Groq..."):
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = QA_PROMPT.format(context=context, question=query)
                answer = llm(prompt)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("🔍 Retrieved Chunks"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**\n{d.page_content[:500]}...")
