from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_DB_PATH = "chroma_db"
HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Check stored documents
retrieved_docs = db.similarity_search("test query", k=5)
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}:\n{doc.page_content}\n{'-'*40}")
