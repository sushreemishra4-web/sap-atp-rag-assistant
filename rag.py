import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a local vector database
client = chromadb.Client()
collection = client.create_collection("sap_docs")

# Your documents — chunks of knowledge
documents = [
    "ATP stands for Available to Promise. It checks if stock is available for a sales order.",
    "In SAP SD, a delivery block prevents a sales order from being delivered to the customer.",
    "Credit management in SAP checks a customer's credit limit before processing an order.",
    "A transfer order in SAP WM is used to move stock from one storage bin to another.",
    "Backorder processing in SAP allows you to reassign stock from lower priority orders to higher ones.",
]

# Embed and store each document
embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))]
)

# Query
query = "stock not in the right place"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("Query:", query)
print("\nTop matches:")
for doc in results["documents"][0]:
    print("-", doc)