# ingest.py
import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pytesseract
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import tempfile
import uuid
import shutil


shutil.rmtree("uploads") if os.path.isdir("uploads") else None

# Helper: OCR a scanned PDF into a single text string (page-level)
def ocr_pdf_to_text(pdf_path: str, dpi=300) -> List[Document]:
    pages = convert_from_path(pdf_path, dpi=dpi)
    docs = []
    for i, page in enumerate(pages):
        with BytesIO() as b:
            page.save(b, format="PNG")
            b.seek(0)
            text = pytesseract.image_to_string(Image.open(b))
            metadata = {"source": pdf_path, "page": i+1}
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_file_to_docs(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    # If PDF has text, prefer PyPDFLoader
    if ext == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()  # this may produce page-level docs
            # If loader produced empty text (scanned), fallback to OCR
            all_text = "".join([d.page_content.strip() for d in docs])
            if len(all_text) < 50:
                print("PDF appears scanned or has little text, running OCR...")
                docs = ocr_pdf_to_text(file_path)
        except Exception as e:
            print("PyPDFLoader failed, running OCR:", e)
            docs = ocr_pdf_to_text(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding='utf8')
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Ensure metadata has source & chunk ids later
    for d in docs:
        if "source" not in d.metadata:
            d.metadata["source"] = file_path
    return docs

def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_docs = []
    for doc in docs:
        pieces = splitter.split_documents([doc])
        # add page metadata if present
        new_docs.extend(pieces)
    return new_docs

def build_faiss_index(docs: List[Document], persist_dir: str):
    if os.path.exists("faiss_index"):
        if os.path.isfile("faiss_index"):
            os.remove("faiss_index")
        elif os.path.isdir("faiss_index"):
            shutil.rmtree("faiss_index")
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_dir):
        # load existing and add
        index = FAISS.load_local(persist_dir, embeddings)
        index.add_documents(docs)
    else:
        index = FAISS.from_documents(docs, embeddings)
        index.save_local(persist_dir)
    return index

# Convenience high-level ingest function
def ingest_files(file_paths: List[str], persist_dir: str = "faiss_index"):
    all_docs = []
    for path in file_paths:
        docs = load_file_to_docs(path)
        print(f"Loaded {len(docs)} docs from {path}")
        docs = split_documents(docs)
        # augment metadata with chunk_id
        for i, d in enumerate(docs):
            if "chunk_id" not in d.metadata:
                d.metadata["chunk_id"] = str(uuid.uuid4())
        all_docs.extend(docs)
    print("Total chunks:", len(all_docs))
    index = build_faiss_index(all_docs, persist_dir)
    print("Index saved to", persist_dir)
    return index
