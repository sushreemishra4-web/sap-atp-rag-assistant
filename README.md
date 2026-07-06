# SAP ATP RAG Assistant

A document-grounded Q&A assistant built with Python, ChromaDB, and Groq.
Ask questions about SAP ATP (Available to Promise) and get answers sourced directly from your documentation.

## How it works
- SAP ATP document is chunked and embedded using SentenceTransformers
- Chunks are stored in a local ChromaDB vector database
- User questions are semantically matched to relevant chunks
- Groq LLM generates grounded answers from retrieved context only

## Tech stack
- Python 3.11
- Streamlit — UI
- ChromaDB — vector database
- SentenceTransformers — embeddings
- Groq API — LLM (Llama 3.3 70b)

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies
3. Create a `.env` file with your Groq API key:
GROQ_API_KEY=your-key-here
4. Run the app:
streamlit run app.py