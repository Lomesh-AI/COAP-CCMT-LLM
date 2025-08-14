# this script will load and convert chunks into chunks embedding (the whole chunk)
# then will store these embedding in the chroma vector store
# into numerical representation

# we use vector store as they are designed for semantic searches


import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from data_prepare import load_documents_and_split
from dotenv import load_dotenv
load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/chroma_db')

def create_vector_store():
    """
    this script will load and convert chunks into chunks embedding (the whole chunk)
    then will store these embedding in the chroma vector store
    """

    chunks = load_documents_and_split()
    print('loading and embedding chunks......')

    embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = VECTOR_STORE_PATH
    )

    vector_store.persist()
    print("Vector store created and saved successfully")
    
if __name__ ==  '__main__':
    create_vector_store()
    


    