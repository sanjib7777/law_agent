from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.gemini_api_key
)

def ask_gemini(query: str, context_chunks: list[str], metadatas: list[dict]):
    context_with_sources = []
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas), 1):
        # Build a source string from metadata fields
        filename = meta.get("filename", "unknown_file")
        part = meta.get("part")
        chapter = meta.get("chapter")
        chunk_index = meta.get("chunk_index")
        source_parts = [f"filename={filename}"]
        if part:
            source_parts.append(f"part={part}")
        if chapter:
            source_parts.append(f"chapter={chapter}")
        if chunk_index is not None:
            source_parts.append(f"chunk={chunk_index}")
        source = ", ".join(source_parts)
        context_with_sources.append(f"[Source: {source}]\n{chunk}")
    context_text = "\n\n".join(context_with_sources)

    print(context_text)

    prompt = f"""
    You are an expert assistant for legal documents.

    Use ONLY the context below to answer the question.
    If the context does not contain the answer, say:
    "No relevant information found in the uploaded document."

    --- CONTEXT ---
    {context_text}
    ----------------

    QUESTION: {query}

    Provide a clear and concise answer. When you use information from the context, cite the source in your answer using the [Source: ...] notation as provided above.
    """

    response = llm.invoke(prompt)
    return response.content
