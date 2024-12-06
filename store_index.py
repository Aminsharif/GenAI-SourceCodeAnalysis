from src.helper import repo_ingestion, load_repo, text_split, load_embedding
from langchain.vectorstores import Chroma
import os


data = load_repo("repo/")
documents = text_split(data)
embeddings = load_embedding()

#storing vector in choramdb
vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory='./db')
vectordb.persist()