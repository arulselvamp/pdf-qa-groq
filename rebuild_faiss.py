# rebuild_faiss.py
import os, json
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_MODEL = "all-MiniLM-L6-v2"
SAVE_DIR = "faiss_index"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)

os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for doc in chunks:
        f.write(json.dumps({"page_content": doc.page_content, "metadata": getattr(doc, "metadata", {})}, ensure_ascii=False) + "\n")

embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(SAVE_DIR)
print("Rebuilt FAISS and saved to", SAVE_DIR)
