# agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

print('going inside')

def ask_gemini(query: str, context_chunks: list[str]):
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
    You are an expert assistant for legal documents.

    Use ONLY the context below to answer the question.
    If the context does not contain the answer, say:
    "No relevant information found in the uploaded document."

    --- CONTEXT ---
    {context_text}
    ----------------

    QUESTION: {query}

    Provide a clear and concise answer.
    """
    
    messages = prompt.format_messages(
        context=context_text,
        query=query
    )

    response = llm(messages)
    return response.content
