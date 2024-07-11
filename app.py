import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import requests
from bs4 import BeautifulSoup
import os
import gradio as gr


# Step 1: Define PromptTemplate class using LangChain's format
class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        formatted_text = self.template
        for key, value in kwargs.items():
            formatted_text = formatted_text.replace("{" + key + "}", str(value))
        return formatted_text


# Step 2: Load embedding model and tokenizer
embedding_model_name = "ls-da3m0ns/bge_large_medical"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)
embedding_model.eval()  # Set model to evaluation mode

# Move the embedding model to GPU
device = torch.device("cuda")
embedding_model.to(device)

# Step 3: Load Faiss index
index_file = "faiss_index.index"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    assert isinstance(index, faiss.IndexFlat), "Expected Faiss IndexFlat type"
    assert index.d == 1024, f"Expected index dimension 1024, but got {index.d}"
else:
    raise ValueError(f"Faiss index file '{index_file}' not found.")

# Step 4: Prepare URLs
urls_file = "crawled_urls.txt"
if os.path.exists(urls_file):
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f]
else:
    raise ValueError(f"URLs file '{urls_file}' not found.")

# Step 5: Check if sample embeddings file exists, if not create it
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
    ]

    sample_embeddings = []
    for text in sample_texts:
        inputs = embedding_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            sample_embeddings.append(embedding)

    sample_embeddings = np.vstack(sample_embeddings)
    np.save(sample_embeddings_file, sample_embeddings)
else:
    sample_embeddings = np.load(sample_embeddings_file)


# Step 6: Define function for similarity search
def search_similar(query_text, top_k=3):
    inputs = embedding_tokenizer(query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    _, idx = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        key = int(idx[0][i])
        results.append(urls[key])  # Return URLs only for simplicity

    return results


# Step 7: Function to extract content from URLs
def extract_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example: Extracting relevant content based on query
        paragraphs = soup.find_all('p')
        relevant_content = ""
        for para in paragraphs:
            relevant_content += para.get_text().strip()

        return relevant_content.strip()  # Return relevant content as a single string
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""


# Step 8: Use the LangChain text generation pipeline for generating answers
generation_model_name = "microsoft/Phi-3-mini-4k-instruct"
text_generator = pipeline("text-generation", model=generation_model_name, device=0)


# Step 9: Function to generate answer based on query and content
def generate_answer(query, contents):
    answers = []
    prompt_template = PromptTemplate("""
    
    ### Medical Assistant Context ###
As a helpful medical assistant, I'm here to assist you with your query.

### Medical Query ###
Query: {query}

### Explanation ###
{generated_text}

### Revised Response ###
Response: {generated_text}
""")

    for content in contents:
        if content:
            prompt = prompt_template.format(query=query, content=content, generated_text="")
            # Ensure prompt is wrapped in a list for text generation
            generated_texts = text_generator([prompt], max_new_tokens=200, num_return_sequences=1, truncation=True)

            # Ensure generated_texts is a list and not None
            if generated_texts and isinstance(generated_texts, list) and len(generated_texts) > 0:
                # Extract the response text only from the generated result
                response = generated_texts[0][0]["generated_text"]
                response_start = response.find("Response:") + len("Response:")
                answers.append(response[response_start:].strip())
            else:
                answers.append("No AI-generated text found.")
        else:
            answers.append("No content available to generate an answer.")
    return answers


# Gradio interface
def process_query(query):
    top_results = search_similar(query, top_k=3)
    if top_results:
        content = extract_content(top_results[0])
        answer = generate_answer(query, [content])[0]

        response = f"Rank 1: URL - {top_results[0]}\n"
        response += f"Generated Answer:\n{answer}\n"

        similar_urls = "\n".join(top_results[1:])  # The second and third URLs as similar URLs
        return response, similar_urls
    else:
        return "No results found.", "No similar URLs found."


demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Similar URLs")
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)