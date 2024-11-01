from langchain.document_loaders import  DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import shutil

os.environ["OPENAI_API_BASE"] ='http://10.35.151.101:8001/v1'
os.environ["OPENAI_API_KEY"] = "sk-1234"
CHROMA_PATH = "docs_embedding"

def load_documents():
  document_loader = PyPDFDirectoryLoader('docs') 
  return document_loader.load()


def split_text(documents: list[Document]):

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # Size of each chunk in characters
    chunk_overlap=200, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
  document = chunks[0]
  print(document.page_content)
  print(document.metadata)

  return chunks


def save_to_chroma(chunks: list[Document]):
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
      chunks,
      OpenAIEmbeddings(),
      persist_directory=CHROMA_PATH
    )

    db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
  documents = load_documents()
  chunks = split_text(documents)
  save_to_chroma(chunks)

print('*** Converting documents into embeddings and creating a vector store(s)')
generate_data_store()
