# to create a fastapi app to serve a langchain model
from fastapi import FastAPI, UploadFile, File, HTTPException
from Upload_into_database.upload_acts import ingest_act_pdfs
from retrieve import legal_rag_answer
from Upload_into_database.upload_constitution import ingest_constitution
from Upload_into_database.upload_old_case import ingest_case_docx
import os
from typing import List

app = FastAPI()

DATASET_DIR = "dataset"



@app.post("/ingest_constitution/")
async def ingest_constitution_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    pdf_path = os.path.join(DATASET_DIR, file.filename)

    if not os.path.exists(pdf_path):
        raise HTTPException(404, f"{file.filename} not found in dataset folder")

    result = ingest_constitution(pdf_path)

    return {
        "status": "success",
        "filename": file.filename,
        "details": result
    }



@app.post("/ingest_old_case/")
async def ingest_case_laws(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "Only DOCX files allowed")

    docx_path = os.path.join(DATASET_DIR, file.filename)

    if not os.path.exists(docx_path):
        raise HTTPException(404, f"{file.filename} not found in dataset folder")

    return ingest_case_docx(docx_path)



@app.post("/upload_acts/")
async def upload_acts(files: List[UploadFile] = File(...)):
    paths = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{file.filename} is not a PDF")

        pdf_path = os.path.join(DATASET_DIR, file.filename)

        if not os.path.exists(pdf_path):
            raise HTTPException(404, f"{file.filename} not found in dataset folder")

        paths.append(pdf_path)

    result = ingest_act_pdfs(paths)

    return {
        "status": "success",
        "files_used": [f.filename for f in files],
        "details": result
    }

# to create an endpoint to query the model
@app.post("/query/")
def query_model(question: str):
    response = legal_rag_answer(question)
    return {"answer": response}



