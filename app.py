import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.title("SAP ATP Assistant")
st.caption("Ask questions about SAP ATP — powered by RAG")

@st.cache_resource
def load_rag():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("atp_docs")
    
    with open("SAP_ATP_BASIC.txt", "r") as f:
        text = f.read()
    
    def chunk_text(text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks
    
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk{i}" for i in range(len(chunks))]
    )
    
    return embedding_model, groq_client, collection

embedding_model, groq_client, collection = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about SAP ATP...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    retrieved_chunks = "\n".join(results["documents"][0])

    stream = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"""You are an SAP ATP expert.
    Answer using only the context below.
    If the answer is not in the context say so clearly.

    Context:
    {retrieved_chunks}"""},
            *st.session_state.messages
        ],
        model="llama-3.3-70b-versatile",
        stream=True
    )

    with st.chat_message("assistant"):
        reply = ""
        placeholder = st.empty()
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                placeholder.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.expander("Retrieved chunks"):
        for i, chunk in enumerate(results["documents"][0]):
            st.caption(f"Chunk {i+1}")
            st.write(chunk)