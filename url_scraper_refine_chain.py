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
import pandas as pd
from pathlib import Path as p


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
    question_prompt_template = """
                    Please provide a summary of the following text.
                    TEXT: {text}
                    SUMMARY:
                    """

    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["text"]
    )

    refine_prompt_template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
                """

    refine_prompt = PromptTemplate(
        template=refine_prompt_template, input_variables=["text"]
    )
    summary_chain = load_summarize_chain(llm,chain_type="refine",
                                         question_prompt=question_prompt,
                                         refine_prompt=refine_prompt,
                                         return_intermediate_steps=True)
    url_and_summaries = {}
    for doc_file in glob.glob(docs_folder + "/*"):
        loader = PyPDFLoader(doc_file)
        docs = loader.load_and_split()
        refine_outputs = summary_chain({"input_documents": docs})
        final_refine_data = []
        for doc, out in zip(
            refine_outputs["input_documents"], refine_outputs["intermediate_steps"]
        ):
            output = {}
            output["file_name"] = p(doc.metadata["source"]).stem
            output["file_type"] = p(doc.metadata["source"]).suffix
            output["page_number"] = doc.metadata["page"]
            output["chunks"] = doc.page_content
            output["concise_summary"] = out
            final_refine_data.append(output)
        pdf_refine_summary = pd.DataFrame.from_dict(final_refine_data)
        pdf_refine_summary = pdf_refine_summary.sort_values(
            by=["file_name", "page_number"]
        )  # sorting the datafram by filename and page_number
        pdf_refine_summary.reset_index(inplace=True, drop=True)
        print(pdf_refine_summary["concise_summary"])
        url_and_summaries[doc_file] = '\n'.join(list(pdf_refine_summary["concise_summary"].unique()))
    
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
