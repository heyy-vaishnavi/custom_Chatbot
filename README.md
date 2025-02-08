#Brainlox Technical Courses Chatbot#

This project implements a chatbot designed to answer questions specifically about the technical courses offered on Brainlox (https://brainlox.com/courses/category/technical). It leverages a combination of cutting-edge technologies, including LangChain for orchestration, Hugging Face embeddings for semantic understanding, ChromaDB for efficient information retrieval, and the GPT4All Large Language Model (LLM) for generating human-like responses.
![chatbot](https://github.com/user-attachments/assets/b2adf8c9-a3fd-4baa-a193-a5c79da1518d)

#Installation#

1.Create a virtual environment
2.Install the required packages
3.Set up environment variables

#Usage#

1.Index the data:
Bash
python data_loader.py
This script fetches the content from the Brainlox website, processes it, creates embeddings, and stores them in the ChromaDB.  It needs to be run only once (or when you want to update the indexed data).
2.Run the Flask app:
Bash
python app.py
3.Open your web browser and go to http://127.0.0.1:5000 to interact with the chatbot.

#Features#
1.Targeted Question Answering
2.Persistent Knowledge Base
3.Semantic Understanding
4.Natural Language Responses
5.Easy-to-use Web Interface
6.Source Document Tracking
