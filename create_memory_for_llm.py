from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

# data load
DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#create chunks of text from documents

def create_chunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks


#create vector embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


#store embedding in faiss
DB_FAISS_PATH="vectorstore/db_faiss"

def create_vector_db():
    documents=load_pdf_files(DATA_PATH)
    text_chunks=create_chunks(documents)
    embedding_model=get_embedding_model()
    db=FAISS.from_documents(text_chunks,embedding_model)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()