from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()
def process_pdf():

    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    
    
    loader = PyPDFLoader("const.pdf")
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    print(f"Loaded {len(docs)} documents from the PDF.")    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    print ("Splitting text into chunks...")
    chunk = text_splitter.split_text(full_text)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        
    )
    url = "http://localhost:6333"
    print("Connecting to Qdrant at", url)
    client = QdrantClient(url=url)
    if not client.collection_exists("Law_Docs"):
        client.create_collection(
            collection_name="Law_Docs",
            vectors_config=VectorParams(
                size=1024,           # must match embedding dimension
                distance=Distance.COSINE
            )
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="Law_Docs",
        embedding=embeddings
    )
    vector_store.add_texts(chunk)
    print("Vector store created with collection name 'Law_Docs'")
    return vector_store


