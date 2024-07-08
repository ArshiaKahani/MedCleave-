import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
import pickle
import os

# Set environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Your script continues here...

def delete_files(*files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def crawl_drugbank_urls(base_url, start_index, end_index, urls_file, contents_file):
    urls = []
    contents = []

    for i in range(start_index, end_index + 1):
        # Construct the URL with the correct format
        url = f"{base_url}{str(i).zfill(5)}"

        response = requests.get(url)
        if response.status_code == 200:
            print(f"Successfully retrieved {url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text(separator=' ')
            urls.append(url)
            contents.append(content)

            # Save progress after each successful fetch
            with open(urls_file, 'a') as uf, open(contents_file, 'wb') as cf:
                uf.write(url + '\n')
                pickle.dump(contents, cf)
        else:
            print(f"Failed to retrieve {url}")

    return urls, contents

def get_bert_embeddings(texts, model, tokenizer):
    embeddings = []
    for i, text in enumerate(texts, 1):
        print(f"Computing BERT embeddings for text {i} out of {len(texts)}")
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()[0])  # Use [0] to get the first token's embedding
    return embeddings

def index_embeddings(embeddings, index_file):
    dimension = embeddings[0].shape[0]  # Adjust dimension to match the shape
    index = faiss.IndexFlatL2(dimension)

    # Normalize embeddings
    embeddings = np.array([emb / np.linalg.norm(emb) for emb in embeddings])

    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

    return index

def save_urls_to_file(urls, filename):
    with open(filename, 'w') as file:
        for url in urls:
            file.write(url + '\n')
            print(f"Saved URL: {url}")

def search_similar_urls(query_embedding, index, urls):
    # Perform similarity search
    distances, indices = index.search(query_embedding.reshape(1, -1), k=len(urls))

    print("Similar URLs:")
    for i, idx in enumerate(indices[0]):
        print(f"URL: {urls[idx]}")
        print(f"Distance: {distances[0][i]}")

base_url = "https://go.drugbank.com/drugs/DB"
start_index = 1
end_index = 100  # Adjust as needed

urls_file = 'crawled_urls.txt'
contents_file = 'crawled_contents.pkl'
index_file = 'faiss_index.index'

try:
    # Delete previous files
    print("Deleting previous files...")
    delete_files(urls_file, contents_file, index_file)
    print("Deleted previous files.")

    # Crawl the URLs
    print(f"Crawling DrugBank URLs from {base_url}00001 to {base_url}{str(end_index).zfill(5)}...")
    urls, contents = crawl_drugbank_urls(base_url, start_index, end_index, urls_file, contents_file)
    print("Crawling finished.")

    # Load BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("BERT model and tokenizer loaded.")

    # Get BERT embeddings
    print("Computing BERT embeddings...")
    embeddings = get_bert_embeddings(contents, model, tokenizer)
    print("BERT embeddings computed.")

    # Index embeddings with FAISS
    print("Indexing embeddings with FAISS...")
    index = index_embeddings(embeddings, index_file)
    print(f"Embeddings indexed and saved to {index_file}.")

    # Save the URLs to a text file
    print(f"Saving crawled URLs to {urls_file}...")
    save_urls_to_file(urls, urls_file)
    print("URLs saved.")

    # Example of similarity search
    print("Performing similarity search example...")
    query_index = 0  # Example query index
    query_embedding = embeddings[query_index]

    search_similar_urls(query_embedding, index, urls)

    print("Script execution completed.")

except Exception as e:
    print(f"An error occurred: {str(e)}")