import os
import requests
import logging
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  # Explicit import

# Configuration (better to put these in a .env file or command-line args)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")  # Default if env variable not set
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# Set up logging (do this once at the module level)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_url(url):
    """Fetch HTML, extract text, and chunk it."""
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "aside", "footer"]):  # Add more if needed
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())  # More efficient cleaning

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Number of chunks: {len(chunks)}")

    documents = [Document(page_content=chunk, metadata={"source": url, "chunk": i}) for i, chunk in enumerate(chunks)]
    return documents

def create_embeddings_and_store(documents):
    """Generate embeddings and store them in ChromaDB."""
    if not documents:
        logging.warning("No documents to store.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME) # No need for model_kwargs if device is auto

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    try:
        db = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DB_PATH)
        logging.info("Embeddings successfully stored in ChromaDB.")
        return db
    except Exception as e:
        logging.exception(f"Error creating/storing embeddings: {e}")  # Log the full exception
        return None



if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical" # Or any other URL

    documents = load_data_from_url(url)
    logging.info(f"Loaded {len(documents)} document(s).")

    if documents:
        db = create_embeddings_and_store(documents)
        if db:
            logging.info("ChromaDB created/updated successfully.")