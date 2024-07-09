import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os

# Set environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Step 1: Load the model and tokenizer
model_name = "ls-da3m0ns/bge_large_medical"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# Step 2: Load Faiss index
index_file = "faiss_index.index"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    assert isinstance(index, faiss.IndexFlat), "Expected Faiss IndexFlat type"
    assert index.d == 1024, f"Expected index dimension 1024, but got {index.d}"
else:
    raise ValueError(f"Faiss index file '{index_file}' not found.")

# Step 3: Prepare URLs
urls_file = "crawled_urls.txt"
if os.path.exists(urls_file):
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f]
else:
    raise ValueError(f"URLs file '{urls_file}' not found.")

# Step 4: Check if sample embeddings file exists, if not create it
sample_embeddings_file = "sample_embeddings.npy"
if not os.path.exists(sample_embeddings_file):
    print("Sample embeddings file not found, creating new sample embeddings...")
    # Generate sample data to fit PCA
    sample_texts = [
        "medical diagnosis",
        "healthcare treatment",
        "patient care",
        "clinical research",
        "disease prevention"
    ]  # Replace with more representative samples if available

    sample_embeddings = []
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling over tokens
            sample_embeddings.append(embedding)

    sample_embeddings = np.vstack(sample_embeddings)
    np.save(sample_embeddings_file, sample_embeddings)
else:
    sample_embeddings = np.load(sample_embeddings_file)

# Step 5: Define function for similarity search
def search_similar(query_text, top_k=3):
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
        key = int(idx[0][i])  # Convert key to integer
        reconstructed_embedding = index.reconstruct(int(key)).astype(np.float32)  # Ensure reconstruction type
        similarity_score = np.dot(query_embedding[0], reconstructed_embedding)
        results.append((urls[key], similarity_score))

    return results

# Step 6: User input and search
query = input("Enter your query: ")
top_results = search_similar(query, top_k=3)

# Step 7: Display results
print(f"Top {len(top_results)} results for query: '{query}'")
for rank, (url, similarity_score) in enumerate(top_results, start=1):
    print(f"Rank {rank}: URL - {url}, Similarity Score - {similarity_score:.4f}")