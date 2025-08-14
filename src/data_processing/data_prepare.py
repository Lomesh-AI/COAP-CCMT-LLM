import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_and_split():
    # defining path to raw data/files
    RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw')

    # will contain all characters from all the pdf
    documents = []

    # getting text from all the raw files
    for filename in os.listdir(RAW_DATA_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(RAW_DATA_PATH, filename)
            print('Extracting Text form ', filename)
            loader = PyPDFLoader(file_path=file_path)
            documents.extend(loader.load())

    print('No of documents extracted', len(documents))       

    # now will divide these documents into smaller chunks for RAG
    # IMPORTANT : LLMs have small context window  
    # ns it will first try to split the text by paragraphs (\n\n), 
    # then by single newlines (\n), then by spaces, and only as a last resort by individual characters. This process ensures that, wherever possible, it keeps related sentences and paragraphs together in the same chunk.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents)

    print("Total chunks created: ", len(chunks))

    print('Lenght of first chunk', len(chunks[0].page_content))

    return chunks

if __name__ == "__main__":
    load_documents_and_split()
    # these chunks will be used to create chunk embedding (here we represent whole chunk as a vector) 
    # (wont we using word embeddign as chunk embeeding will have sematic meaning and context of the whole chunk)


