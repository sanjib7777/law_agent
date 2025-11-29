from fastapi import APIRouter, HTTPException
from src.api.dependencies import embedding_service, vector_store

router = APIRouter()

@router.post("/retrieve")
async def retrieve_document(request: str):
    """
    Retrieve relevant documents from vector database
    """
    try:
        query_embedding = await embedding_service.get_embedding(request)

        # Search for similar documents
        contents, scores = await vector_store.search_similar(query_embedding, top_k=3)
        
        # Prepare response
        results = [
            {"content": content}
            for content in contents
        ]
        
        return {"results": results, "scores": scores}
    except Exception as e:
        print("Error during document retrieval:", str(e))
        return HTTPException(status_code=500, detail="Internal Server Error")