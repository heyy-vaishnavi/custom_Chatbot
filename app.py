import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain  # Or RetrievalQA if you don't need sources
from langchain_community.llms import GPT4All
from dotenv import load_dotenv  # For environment variables

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration (from environment variables or defaults)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")  # Make sure to set this in .env
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", 2))  # Number of documents to retrieve


# Load embeddings and ChromaDB
try:
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME) # No model_kwargs needed if using auto device
    if not os.path.exists(CHROMA_DB_PATH):
        logging.error(f"ChromaDB directory '{CHROMA_DB_PATH}' not found. Run indexing first.")
        db = None
    else:
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        logging.info(f"ChromaDB loaded from {CHROMA_DB_PATH}")
except Exception as e:
    logging.exception(f"Error loading ChromaDB: {e}")
    db = None

# Load GPT4All model
try:
    llm = GPT4All(model=LLM_MODEL_PATH,
                  backend="llama",
                  n_threads=4,
                  max_tokens=512)  # Reduced max tokens
    logging.info(f"LLM loaded from {LLM_MODEL_PATH}")

except Exception as e:
    logging.exception(f"Error loading LLM: {e}")
    llm = None

# Create QA chain (only if both DB and LLM are loaded)
qa_chain = None
if db and llm:
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(  # Or RetrievalQA if you don't need sources
        llm=llm,
        chain_type="refine",  # Or "stuff", "map_reduce" as needed
        retriever=db.as_retriever(search_kwargs={"k": K_RETRIEVAL}),  # Adjust k
        return_source_documents=True # To see what documents are being used
    )
    logging.info("RetrievalQA chain initialized.")
else:
    logging.error("RetrievalQA could not be initialized. Check ChromaDB and LLM paths.")


@app.route('/')
def home():
    return "Welcome to the Chatbot API! Use /api/chat to interact."


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(directory="static", path="favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route('/api/chat', methods=['POST'])
def chat():
    if not qa_chain:
        return jsonify({'error': 'RetrievalQA is not initialized'}), 500

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'Missing query'}), 400

    user_query = user_query[:MAX_QUERY_LENGTH]  # Limit query length

    try:
        result = qa_chain.invoke(user_query)  # Use .invoke() – corrected from .__call__()
        response = {
            "answer": result['answer'],
            "sources": result['sources'],  # Add the sources to the response
            "source_documents": [doc.page_content for doc in result['source_documents']] # Add source documents
        }
        return jsonify(response)
    except Exception as e:
        logging.exception("Error during chat interaction:") # Log the full exception
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)