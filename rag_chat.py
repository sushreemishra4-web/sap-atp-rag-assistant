from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
# Setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Vector database
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("sap_docs")

# Your documents
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
print(f"Document loaded — {len(chunks)} chunks created\n")

# Embed and store
embeddings = embedding_model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk{i}" for i in range(len(chunks))]
)

print("Knowledge base ready. Type your question or 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    query_embedding = embedding_model.encode([user_input]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    retrieved_chunks = "\n".join(results["documents"][0])
    print(f"\n--- Retrieved chunks ---")
    for i, chunk in enumerate(results["documents"][0]):
        print(f"Chunk {i+1}: {chunk[:150]}...")
        print("------------------------\n")
        
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"""You are an SAP ATP expert.
Answer using only the context below.
If the answer is not in the context say so clearly.

Context:
{retrieved_chunks}"""},
            {"role": "user", "content": user_input}
        ],
        model="llama-3.3-70b-versatile",
    )

    reply = response.choices[0].message.content
    print(f"AI: {reply}\n")