from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import requests
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob
from pymongo import MongoClient, errors
from gridfs import GridFS, NoFile
import json
import logging
from bson import ObjectId
import requests  # For downloading PDF from URL
import os


# Function to extract href links from a URL
def extract_href_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='btn inline-flex h-9 items-center', href=True)
    return [link['href'] for link in links if link['href'].startswith('https://arxiv.org/pdf')]

# Function to download PDF from a URL
def download_pdf(url, date_str, download_folder=r'C:\Users\307164\Desktop\Huggingface_Paper_Extractor\pdfs'):
    download_folder = os.path.join(download_folder, date_str)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    response = requests.get(url)
    filename = os.path.join(download_folder, url.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# Function to construct the URL
def construct_url(base_url, date_str):
    """
    Constructs a URL with updated date parameters.

    Args:
        base_url (str): The base URL where parameters need to be updated.
        date_str (str): The new date in 'YYYY-MM-DD' format.
    
    Returns:
        str: Updated URL with the new date.
    """
    # Parse the original URL
    parsed_url = urlparse(base_url)
    query_params = parse_qs(parsed_url.query)

    # Update the 'date' parameter
    query_params['date'] = [date_str]  # Update this to change the date

    # Reconstruct the URL with the new parameters
    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, new_query, parsed_url.fragment))
    
    return new_url

def extract_matching_urls(url_list):
    matching_urls = []
    for url in url_list:
        if url.startswith('/papers/') and url[8:].replace('.', '').isdigit():
            matching_urls.append('https://huggingface.co' + url)
    return matching_urls

def summarize_docs_from_folder(docs_folder):
    # Map Prompt Template
    map_prompt = """
    Provide a comprehensive summary of the research paper "{text}". Your summary should include the following aspects:
    - Problem Statement: Clearly articulate the main problem or research question addressed in the paper. (Limit: 50-100 words)
    - Methodology: Describe the overall methodology or approach used by the authors to tackle the problem. (Limit: 50-100 words)
    - Experimental Procedures: Summarize the experimental setup and procedures followed in the study. (Limit: 50-100 words)
    - Key Findings: Highlight the main results or outcomes obtained from the research. (Limit: 50-100 words)
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    # Combine Prompt Template
    combine_prompt = """
    Compose a detailed summary of the research paper represented by the given text enclosed in triple backquotes. Your summary should cover the following aspects:
    - Problem Addressed: Briefly explain the problem or research question the paper aims to solve. (Limit: 50-100 words)
    - Approach Taken: Describe the methodology or approach employed by the authors to address the problem. (Limit: 50-100 words)
    - Experiments Conducted: Summarize the experiments, including any data collection or analysis methods used. (Limit: 50-100 words)
    - Outcomes Obtained: Discuss the main findings or outcomes of the research, including any significant results or conclusions. (Limit: 50-100 words)
    Please ensure to provide the author's name, title of the paper, associated field, and classification from a use case perspective.
    ```{text}```

    DETAILED SUMMARY:
    """

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                        #max_new_tokens=3000
                                        verbose=True
                                        )
    url_and_summaries = {}
    for doc_file in glob.glob(docs_folder + "/*"):
        loader = PyPDFLoader(doc_file)
        docs = loader.load_and_split()
        summary = summary_chain.run(docs)
        print("Summary for:", doc_file)
        print(summary)
        print("\n")
        url_and_summaries[doc_file] = summary
    
    return url_and_summaries


base_url = "https://huggingface.co/papers"
date_str = "2024-03-15"  # Change this to the desired date
updated_url = construct_url(base_url, date_str)
print(updated_url)


response = requests.get(updated_url)
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

links = soup.find_all('a', href=True)

urls = [link['href'] for link in links]

matched_urls = extract_matching_urls(urls)

for url in matched_urls:
    print(url)


for url in matched_urls:
    href_links = extract_href_links(url)
    for link in href_links:
        print(link)
        download_pdf(link, date_str=date_str)


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", 
             temperature=0.2)


pdfs_folder = r"C:\Users\307164\Desktop\Huggingface_Paper_Extractor\pdfs\{}".format(date_str)
url_and_summaries = summarize_docs_from_folder(pdfs_folder)


folder_path = f"inference/{date_str}"
os.makedirs(folder_path, exist_ok=True)

with open(f"{folder_path}/inference.json", "w") as f:
    json.dump(url_and_summaries, f)
