from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv(verbose=True) 

CHROMA_PATH = os.getenv('CHROMA_PATH','./data/embedding')
DOCUMENT_DIR = os.getenv('DOCUMENT_DIR','./data/docs')
INFERENCE_DEVICE = os.getenv('INFERENCE_DEVICE','cpu')
MODEL_EMBEDDINGS = os.getenv('MODEL_EMBEDDINGS','sentence-transformers/all-MiniLM-L6-v2')

def load_documents():
  document_loader = DirectoryLoader(DOCUMENT_DIR) 
  return document_loader.load()


def split_text(documents: list[Document]):

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # Size of each chunk in characters
    chunk_overlap=300, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
  document = chunks[0]
  #print(document.page_content)
  #print(document.metadata)

  return chunks


def save_to_chroma(chunks: list[Document]):
  model_kwargs = {"device": INFERENCE_DEVICE}
  encode_kwargs = {"normalize_embeddings": False}
  hf = HuggingFaceEmbeddings(
      model_name=MODEL_EMBEDDINGS, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
  )
  db = Chroma.from_documents(
    chunks,
    hf,
    persist_directory=CHROMA_PATH
  )

  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
  print('*** Converting documents into embeddings and creating a vector store(s)')
  documents = load_documents()
  chunks = split_text(documents)
  chromadb.api.client.SharedSystemClient.clear_system_cache()
  save_to_chroma(chunks)
  print('*** Documents embeddings created in vector store(s)')
  
