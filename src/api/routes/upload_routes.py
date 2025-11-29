from fastapi import APIRouter, UploadFile, File, HTTPException

from src.api.dependencies import embedding_service, vector_store, document_processor
from src.services.file_parser import FileParser


router = APIRouter()



@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload the pdf file and store it in vector database
    """
    try:
        
        # Parse file content based on file type
        text_content = await FileParser.parse_file(file, file.filename)
        
        # Process document
        chunks = document_processor.process_document(text_content, file.filename)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content found in document")
        
        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await embedding_service.get_embeddings(chunk_texts)
        
        # Store in vector database
        document_id = await vector_store.store_documents(chunks, embeddings)
        
        return {
            "message": "Document ingested successfully",
            "chunks_processed": len(chunks),
            "document_id":document_id
        }
        
    except Exception as e:
        print("Error during file upload:", str(e))
        return HTTPException(status_code=500, detail="Internal Server Error")
