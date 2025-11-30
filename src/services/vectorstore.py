from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Tuple
from uuid import uuid4

from src.config.settings import settings

class VectorStoreService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_url,
            port=settings.qdrant_port
        )
        self.collection = settings.qdrant_collection
        self._ensure_collection()


    def _ensure_collection(self):
        """
        Ensure that the Qdrant collection exists; create it if it doesn't.
        """
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection not in collection_names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
    

    async def store_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> str:
        """
        Store documents and their embeddings in Qdrant
        """
        points = []
        document_id = str(uuid4())

        for i, (doc, embeddings) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=uuid4().int & ((1 << 64) - 1),
                vector=embeddings,
                payload={
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "document_id": document_id
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

        return document_id
    

    async def search_similar(self, query_embedding: List[float], top_k: int = None) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for similar documents
        """
        if top_k is None:
            top_k = 5
        
        search_result = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k
        )
        # Initialize lists for the results
        contents = []
        scores = []
        metadatas = []

        # Process each hit
        for hit in search_result:
            label = hit[0]  
            scored_points = hit[1]
            for scored_point in scored_points:
                score = scored_point.score
                payload = scored_point.payload
                content = payload.get("content", "")
                metadata = payload.get("metadata", {})
                contents.append(content)
                scores.append(score)
                metadatas.append(metadata)
        return contents, scores, metadatas