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

def download_pdf(url, date_str, download_folder=r'C:\Users\307164\Desktop\Huggingface_Paper_Extractor\arxiv_pdfs'):
    download_folder = os.path.join(download_folder, date_str)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    response = requests.get(url)
    filename = os.path.join(download_folder, url.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

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

response = requests.get("https://arxiv.org/search/?query=large+language+models&searchtype=all&abstracts=show&order=-announced_date_first&size=50")
date_str = "2024-03-17"   
#response = requests.get(f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=large+language+models&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={date_str}&date-to_date={date_str}&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first")
soup = BeautifulSoup(response.text, 'html.parser')
# Find all 'a' tags
# Find all 'a' tags
links = soup.find_all('a')

# Extract href URLs starting with 'https://arxiv.org/pdf'
pdf_links = [link.get('href') for link in links if link.get('href') and link.get('href').startswith('https://arxiv.org/pdf')]

#for link in pdf_links:
#    download_pdf(link, 
#                date_str=date_str)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", 
             temperature=0.2)

pdfs_folder = r"C:\Users\307164\Desktop\Huggingface_Paper_Extractor\arxiv_pdfs\{}".format(date_str)
url_and_summaries = summarize_docs_from_folder(pdfs_folder)


folder_path = f"inference/{date_str}"
os.makedirs(folder_path, exist_ok=True)

with open(f"{folder_path}/arxiv_inference.json", "w") as f:
    json.dump(url_and_summaries, f)
