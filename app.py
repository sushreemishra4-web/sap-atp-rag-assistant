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
st.success(f"Knowledge base ready")

question = st.text_input("Your question:", placeholder="What is ATP in SAP?")

if st.button("Ask") and question:
    with st.spinner("Searching knowledge base..."):
        query_embedding = embedding_model.encode([question]).tolist()
        results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    retrieved_chunks = "\n".join(results["documents"][0])

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"""You are an SAP ATP expert.
Answer using only the context below.
If the answer is not in the context say so clearly.

Context:
{retrieved_chunks}"""},
            {"role": "user", "content": question}
        ],
        model="llama-3.3-70b-versatile",
    )

    reply = response.choices[0].message.content
    
    st.markdown("### Answer")
    st.write(reply)
    
    with st.expander("Retrieved chunks"):
        for i, chunk in enumerate(results["documents"][0]):
            st.caption(f"Chunk {i+1}")
            st.write(chunk)