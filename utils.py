# utils.py
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import pdfplumber

def load_resume(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("data/vector_store")
    return vectorstore

def extract_text_from_pdf(file) -> str:
    """Extracts all text from a PDF file-like object."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
