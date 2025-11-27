# to create a fastapi app to serve a langchain model
from fastapi import FastAPI, UploadFile, File
from upload import process_pdf
from retrive import query_response
from agent import ask_vibethinker

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/upload/")
def upload_file():
    result = process_pdf()
    return {
        "message": "File processed successfully",
        "chunks_stored": result["chunks_stored"]
    }

    
# to create an endpoint to query the model
@app.post("/query/")
def query_model(question: str):
    response = query_response(question)
    print("retrive response")
    result = ask_vibethinker(question, [doc.page_content for doc in response])
    return {"answer": result}



