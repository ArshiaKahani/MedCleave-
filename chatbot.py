import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# Step 1: Load the model and tokenizer
model_name = "ls-da3m0ns/bge_large_medical"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 2: Prepare example data (replace with your actual data)
urls = ["crawled_urls.txt"]

# Example embeddings (replace with actual embeddings)
embeddings = np.random.rand(len(urls), 768)  # Replace with actual embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]  # Normalize embeddings

# Step 3: Build Faiss index
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = Inner Product (for cosine similarity)
index.add(embeddings.astype(np.float32))

# Step 4: Define function for similarity search
def search_similar(query_text, top_k=5):
    # Tokenize and get embeddings for the query
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling over tokens

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search in Faiss index
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    _, idx = index.search(query_embedding, top_k)

    # Retrieve results
    results = []
    for i in range(top_k):
        results.append((urls[idx[0][i]], np.dot(query_embedding[0], embeddings[idx[0][i]])))

    return results

# Step 5: Example usage
query = "medical condition diagnosis"
top_results = search_similar(query, top_k=5)

# Step 6: Display results
print(f"Top {len(top_results)} results for query: '{query}'")
for rank, (url, distance) in enumerate(top_results, start=1):
    print(f"Rank {rank}: URL - {url}, Similarity Score - {distance:.4f}")
