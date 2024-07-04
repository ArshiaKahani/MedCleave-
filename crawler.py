import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from pdfminer.high_level import extract_text
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
from pdfminer.pdfparser import PDFSyntaxError

# Global counters for URLs and PDFs crawled
total_urls_crawled = 0
total_pdfs_crawled = 0

# Set of visited URLs to prevent duplicates
visited_urls = set()

# Function to get all PDF links from a webpage with retry mechanism
@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry up to 3 times with a 2-second wait between retries
def get_pdf_links(url, domain):
    global total_urls_crawled
    pdf_links = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=50)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.pdf'):
                full_url = urljoin(domain, href)
                pdf_links.append(full_url)
        total_urls_crawled += 1
        print(f"\rURLs crawled: {total_urls_crawled}", end='')  # Print live count without newline
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"HTTP 404 Error: {e}")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
    return pdf_links

# Function to normalize and validate URLs
def normalize_url(url, domain):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = urljoin(domain, url)
    if url.startswith(domain):
        return url
    return None

# Function to recursively get all pages and collect PDF links with retry mechanism
@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry up to 3 times with a 2-second wait between retries
def crawl_site(base_url, domain, depth=0, max_depth=3):
    global total_pdfs_crawled
    if depth > max_depth or base_url in visited_urls:
        return []
    visited_urls.add(base_url)

    pdf_links = get_pdf_links(base_url, domain)

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
            results = executor.map(lambda url: crawl_site(url, domain, depth+1, max_depth), links_to_crawl)
            for result in results:
                pdf_links.extend(result)

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"HTTP 404 Error: {e}")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.RequestException as e:
        print(f"Error accessing {base_url}: {e}")

    return pdf_links

# Function to download a PDF with retry mechanism
@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry up to 3 times with a 2-second wait between retries
def download_pdf(url, save_path):
    global total_pdfs_crawled
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        total_pdfs_crawled += 1
        print(f"\rPDFs crawled: {total_pdfs_crawled}", end='')  # Print live count without newline
    except requests.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Function to extract text from a PDF using PDFMiner.six with specified encoding
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        text = extract_text(pdf_path, codec='utf-8')
    except PDFSyntaxError as e:
        print(f"PDFSyntaxError: {e}")
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

# Function to convert text to BERT embeddings
def convert_text_to_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average pool last layer's output

    return embeddings

# Main process
def main():
    global total_urls_crawled, total_pdfs_crawled
    domain = 'https://www.ema.europa.eu'
    start_url = 'https://www.ema.europa.eu/en/homepage'
    pdf_links = crawl_site(start_url, domain)
    print(f"\n\nFound {total_urls_crawled} URLs and crawled {total_pdfs_crawled} PDFs.")

    # Directory to save PDFs temporarily
    os.makedirs('venv/pdfs', exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    pdf_embeddings = []
    for idx, pdf_url in enumerate(pdf_links):
        pdf_path = f"pdfs/document_{idx}.pdf"
        download_pdf(pdf_url, pdf_path)
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            pdf_embedding = convert_text_to_bert_embeddings(pdf_text, tokenizer, model)
            pdf_embeddings.append(pdf_embedding)

    # Convert embeddings to FAISS index
    if pdf_embeddings:
        vectors = np.vstack(pdf_embeddings)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # Store the index
        faiss.write_index(index, "faiss_index.bin")

        print("FAISS index created and stored successfully.")
    else:
        print("No PDFs processed or extracted.")

if __name__ == "__main__":
    main()
