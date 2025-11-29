from openai import AsyncClient
from typing import List
from src.config.settings import settings

class EmbeddingService:
    def __init__(self):
        self.client = AsyncClient(api_key=settings.openai_api_key)
        self.model = settings.embedding_model


    async def get_embeddings(self, texts: List[str]) -> List[list[float]]:
        """
        Generate embeddings for a list of texts
        """
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
        
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []