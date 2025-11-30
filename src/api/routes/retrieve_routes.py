from fastapi import APIRouter, HTTPException
from src.api.dependencies import embedding_service, vector_store
from src.services.llm_agent import ask_gemini

router = APIRouter()

@router.post("/retrieve")
async def retrieve_document(request: str):
    """
    Retrieve relevant documents from vector database
    """
    try:
        query_embedding = await embedding_service.get_embedding(request)

        # Search for similar documents
        contents, scores, metadatas = await vector_store.search_similar(query_embedding, top_k=3)

        # Prepare response
        results = [
            {"content": content, "metadata": metadata}
            for content, metadata in zip(contents, metadatas)
        ]

        # Use LLM to format the final answer, passing metadata as well
        answer = ask_gemini(request, contents, metadatas)
        return {"answer": answer, "results": results, "scores": scores}
    except Exception as e:
        print("Error during document retrieval:", str(e))
        return HTTPException(status_code=500, detail="Internal Server Error")