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
    

    async def search_similar(self, query_embedding: List[float], top_k: int = None) -> Tuple[List[str], List[float]]:
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

        # Process each hit
        for hit in search_result:
        
            # Since hit is a tuple, the first element (hit[0]) is a label, and the second element (hit[1]) is a list of ScoredPoint objects
            label = hit[0]  
            scored_points = hit[1]
            
            # Process each ScoredPoint in the list
            for scored_point in scored_points:
                # Extract the score and payload from the ScoredPoint
                score = scored_point.score
                payload = scored_point.payload

                # Extract the content from the payload (assuming payload is a dictionary with 'content' key)
                content = payload.get("content", "")  # Using .get() to avoid KeyError if 'content' is missing

                # Append the content and score to their respective lists
                contents.append(content)
                scores.append(score)
        
        return contents, scores