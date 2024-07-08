import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
import time
from ratelimit import limits, sleep_and_retry
import threading

# Global counters for URLs and FAISS index initialization
total_urls_crawled = 0
index_file = 'faiss_index.bin'  # FAISS index file path

# Set of visited URLs to prevent duplicates
visited_urls = set()

# Directory to save crawled URLs
urls_dir = 'crawled_urls'
os.makedirs(urls_dir, exist_ok=True)
urls_file = os.path.join(urls_dir, 'crawled_urls.txt')

# Initialize FAISS index
def initialize_faiss_index(dimension):
    if os.path.exists(index_file):
        os.remove(index_file)
        print("Deleted previous FAISS index file.")
    index = faiss.IndexFlatL2(dimension)
    return index

# Initialize or load FAISS index
dimension = 768  # Dimension of BERT embeddings
index = initialize_faiss_index(dimension)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Lock for thread-safe update of total_urls_crawled
lock = threading.Lock()

# Function to update and print live count of crawled URLs
def update_live_count():
    global total_urls_crawled
    while True:
        with lock:
            print(f"\rURLs crawled: {total_urls_crawled}", end='')
        time.sleep(1)  # Update every second

# Start live count update thread
live_count_thread = threading.Thread(target=update_live_count, daemon=True)
live_count_thread.start()

# Function to save crawled URLs to a file
def save_crawled_urls(url):
    with open(urls_file, 'a') as f:
        f.write(f"{url}\n")
        f.flush()  # Flush buffer to ensure immediate write
        os.fsync(f.fileno())  # Ensure write is flushed to disk

# Function to get all links from a webpage with retry mechanism and rate limiting
@retry(stop_max_attempt_number=3, wait_fixed=2000)
@sleep_and_retry
@limits(calls=10, period=1)  # Adjust calls and period based on website's rate limits
def get_links(url, domain):
    global total_urls_crawled
    links = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=50)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            normalized_url = normalize_url(href, domain)
            if normalized_url and normalized_url not in visited_urls:
                links.append(normalized_url)
                visited_urls.add(normalized_url)
                with lock:
                    total_urls_crawled += 1
                save_crawled_urls(normalized_url)  # Save crawled URL to file

                # Convert text to BERT embeddings and add to FAISS index
                try:
                    text = soup.get_text()
                    if text:
                        embeddings = convert_text_to_bert_embeddings(text, tokenizer, model)
                        index.add(np.array([embeddings]))
                except Exception as e:
                    print(f"Error adding embeddings to FAISS index: {e}")

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"HTTP 404 Error: {e}")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
    return links

# Function to normalize and validate URLs
def normalize_url(url, domain):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = urljoin(domain, url)
    if url.startswith(domain):
        return url
    return None

# Function to recursively get all pages and collect links with retry mechanism and rate limiting
@retry(stop_max_attempt_number=3, wait_fixed=2000)
@sleep_and_retry
@limits(calls=10, period=1)  # Adjust calls and period based on website's rate limits
def crawl_site(base_url, domain, depth=0, max_depth=10):  # Increased max_depth to 10
    if depth > max_depth or base_url in visited_urls:
        return []
    visited_urls.add(base_url)

    links = get_links(base_url, domain)
    print(f"Crawled {len(links)} links from {base_url} at depth {depth}.")  # Debugging info

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links_to_crawl = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            normalized_url = normalize_url(href, domain)
            if normalized_url and normalized_url not in visited_urls:
                links_to_crawl.append(normalized_url)

        with ThreadPoolExecutor(max_workers=500) as executor:
            results = executor.map(lambda url: crawl_site(url, domain, depth + 1, max_depth), links_to_crawl)
            for result in results:
                links.extend(result)

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"HTTP 404 Error: {e}")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.RequestException as e:
        print(f"Error accessing {base_url}: {e}")

    return links

# Function to convert text to BERT embeddings
def convert_text_to_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average pool last layer's output

    return embeddings

# Main process
def main():
    global total_urls_crawled
    domain = 'https://go.drugbank.com/'  # Replace with your new domain
    start_url = 'https://go.drugbank.com/drugs/DB00001'  # Replace with your starting URL


    try:
        # Save the FAISS index at the beginning of the execution
        faiss.write_index(index, index_file)
        print("Initial FAISS index saved.")

        urls = crawl_site(start_url, domain)
        print(f"\n\nFound {total_urls_crawled} URLs.")

        # Save the FAISS index at the end of execution
        faiss.write_index(index, index_file)
        print("Final FAISS index saved.")

    except Exception as e:
        print(f"Exception encountered: {e}")

if __name__ == "__main__":
    main()