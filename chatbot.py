import os
import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

# Set environment variable to avoid OpenMP runtime conflicts (if necessary)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load FAISS index from binary file
index = faiss.read_index('faiss_index.bin')

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def text_to_vector(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings for [CLS] token (first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# Function to get user input
def get_user_input():
    texts = []
    print("Enter text for similarity search (press Enter twice to finish):")
    while True:
        text = input("> ")
        if text.strip() == "":
            break
        texts.append(text)
    return texts

# Example function to perform search
def perform_search(texts, k):
    # Convert texts to vectors
    vectors = np.array([text_to_vector(text) for text in texts])
    # Perform search using FAISS
    D, I = index.search(vectors, k)
    return D, I

# Main function to run the program
def main():
    # Get user input
    texts = get_user_input()

    if not texts:
        print("No input provided. Exiting.")
        return

    k = 3  # Number of nearest neighbors to retrieve

    # Perform search for the provided texts
    distances, indices = perform_search(texts, k)

    # Output results
    for i, text in enumerate(texts):
        print(f"Text: '{text}'")
        print(f"Indices of nearest neighbors: {indices[i]}")
        print(f"Distances of nearest neighbors: {distances[i]}")
        print("\n")

if __name__ == "__main__":
    main()