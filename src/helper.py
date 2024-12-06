import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "/repo"
    Repo.clone_from(repo_url, to_path=repo_path)

def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob="**/*",
                                           suffixes=[".py"],
                                           parser=LanguageParser(language = Language.PYTHON, parser_threshold=500)
                                        )
    data = loader.load()
    return data

def text_split(data):
    spliter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size = 500, 
        chunk_overlap = 20)
    documents = spliter.split_documents(documents=documents)

    return documents

def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings