from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def query_response(question):
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
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

    query = question

    docs = vector_store.similarity_search(query, k=3)
    return docs