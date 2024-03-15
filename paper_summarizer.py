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

llm = OpenAI(temperature=0.2)

def summarize_docs_from_folder(docs_folder):
    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
    #                                      verbose=True
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

# Example usage
pdfs_folder = r"C:\Users\307164\Desktop\Huggingface_Paper_Extractor\pdfs\2024-03-10"
url_and_summaries = summarize_docs_from_folder(pdfs_folder)
date = os.path.basename(pdfs_folder)

# Create the folder if it doesn't exist
folder_path = f"inference/{date}"
os.makedirs(folder_path, exist_ok=True)

with open(f"{folder_path}/inference.json", "w") as f:
    json.dump(url_and_summaries, f)
