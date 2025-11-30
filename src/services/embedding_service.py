
from typing import List
from src.config.settings import settings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class EmbeddingService:
    def __init__(self):
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embedder = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )



    async def get_embeddings(self, texts: List[str]) -> List[list[float]]:
        """
        Generate embeddings for a list of texts using HuggingFaceBgeEmbeddings
        """
        try:
            # HuggingFaceBgeEmbeddings is synchronous, so run in thread executor for async compatibility
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.embedder.embed_documents, texts)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
        
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []