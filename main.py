# to create a fastapi app to serve a langchain model
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
# from Upload_into_database.upload_acts import ingest_act_pdfs
from retrieve import legal_rag_answer
# from Upload_into_database.upload_constitution import ingest_constitution
# from Upload_into_database.upload_old_case import ingest_case_docx
from semantic_cache import get_semantic_cache, set_semantic_cache
from embedding import embeddings
import os
from typing import List
import uuid

app = FastAPI()

DATASET_DIR = "dataset"

@app.middleware("http")
async def add_session_cookie(request: Request, call_next):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    response = await call_next(request)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax"
    )
    return response
@app.get("/") 
def read_root(): 
    return {"message": "Welcome to FastAPI root endpoint!"}

# @app.post("/ingest_constitution/")
# async def ingest_constitution_api(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(400, "Only PDF files allowed")

#     pdf_path = os.path.join(DATASET_DIR, file.filename)

#     if not os.path.exists(pdf_path):
#         raise HTTPException(404, f"{file.filename} not found in dataset folder")

#     result = ingest_constitution(pdf_path)

#     return {
#         "status": "success",
#         "filename": file.filename,
#         "details": result
#     }



# @app.post("/ingest_old_case/")
# async def ingest_case_laws(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(".docx"):
#         raise HTTPException(400, "Only DOCX files allowed")

#     docx_path = os.path.join(DATASET_DIR, file.filename)

#     if not os.path.exists(docx_path):
#         raise HTTPException(404, f"{file.filename} not found in dataset folder")

#     return ingest_case_docx(docx_path)



# @app.post("/upload_acts/")
# async def upload_acts(files: List[UploadFile] = File(...)):
#     paths = []

#     for file in files:
#         if not file.filename.lower().endswith(".pdf"):
#             raise HTTPException(400, f"{file.filename} is not a PDF")

#         pdf_path = os.path.join(DATASET_DIR, file.filename)

#         if not os.path.exists(pdf_path):
#             raise HTTPException(404, f"{file.filename} not found in dataset folder")

#         paths.append(pdf_path)

#     result = ingest_act_pdfs(paths)

#     return {
#         "status": "success",
#         "files_used": [f.filename for f in files],
#         "details": result
#     }

# to create an endpoint to query the model
@app.post("/query/")
async def query_model(question: str, user_id: str, user_role: str):


    # 1️ Try semantic cache
    cached_response = get_semantic_cache(
        session_id=user_id,
        query=question,
        embedder=embeddings
    )

    if cached_response:
        # cached_response must already be a dict
        return cached_response

    # 2 Call RAG pipeline
    response = legal_rag_answer(
        question=question,
        user_id=user_id,
        user_role=user_role
    )

    #  Store full structured response in cache
    set_semantic_cache(
        session_id=user_id,
        query=question,
        answer=response,  
        embedder=embeddings
    )

    # Return response directly
    return response