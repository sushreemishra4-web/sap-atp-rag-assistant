from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("sap_docs")

documents = [
    "SAP_ATP_BASIC.txt",
    "FSCM_Credit_Management_S4HANA.txt",
    "Order_to_Cash_S4HANA.txt"
]

text = ""
for doc in documents:
    with open(doc, "r") as f:
        text += f.read() + "\n\n"

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

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "SAP Knowledge API is running"}

@app.post("/ask")
def ask(payload: Question):
    query_embedding = embedding_model.encode([payload.question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    retrieved_chunks = "\n".join(results["documents"][0])

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"""You are a SAP expert.
Answer using only the context below.
If the answer is not in the context say so clearly.

Context:
{retrieved_chunks}"""},
            {"role": "user", "content": payload.question}
        ],
        model="llama-3.3-70b-versatile",
    )

    return {
        "question": payload.question,
        "answer": response.choices[0].message.content,
        "chunks": results["documents"][0]
    }