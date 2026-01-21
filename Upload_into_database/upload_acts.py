import os
import re
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv
from pypdf import PdfReader
from embedding import embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore



# =========================
# ENV
# =========================
load_dotenv()


COLLECTION_NAME = "nepal_acts"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,timeout=60)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int
):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )



# =========================
# PDF LOADER (PAGE LIMIT SUPPORT)
# =========================
def load_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""

    max_pages = None
    if "Criminal_Procedure_Code_EN.pdf" in pdf_path:
        max_pages = 196

    for i, page in enumerate(reader.pages):
        if max_pages and i >= max_pages:
            break

        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# =========================
# CHAPTER + SECTION EXTRACTION
# =========================
def extract_chapters_and_sections(
    text: str,
    law_name: str,
    document_name: str
) -> List[Dict]:

    chapter_pattern = re.compile(
        r"Chapter\s*[-–]\s*(\d+)\s*\n\s*(.+)",
        re.IGNORECASE
    )

    section_pattern = re.compile(
        r'^\s*(\d+)\.\s*([^:]+):',
        re.MULTILINE
    )

    chunks = []
    chapters = list(chapter_pattern.finditer(text))

    for i, ch in enumerate(chapters):
        start = ch.end()
        end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)

        chapter_text = text[start:end]
        chapter_number = ch.group(1)
        chapter_title = ch.group(2).strip()

        sections = list(section_pattern.finditer(chapter_text))

        for j, sec in enumerate(sections):
            sec_start = sec.end()
            sec_end = sections[j + 1].start() if j + 1 < len(sections) else len(chapter_text)

            section_text = chapter_text[sec_start:sec_end].strip()
            if not section_text:
                continue

            chunks.append({
                "text": section_text,
                "metadata": {
                    "doc_type": "statute",
                    "law_name": law_name,
                    "document_name": document_name,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "section_number": sec.group(1),
                    "section_title": sec.group(2).strip()
                }
            })

    return chunks

def get_vector_store() -> QdrantVectorStore:
    """
    Ensures collection exists and returns vector store
    """
    vector_size = 1024

    ensure_collection_exists(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        vector_size=vector_size
    )

    return QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )


# =========================
# INGEST FROM API (MAIN ENTRY)
# =========================
def ingest_act_pdfs(pdf_paths: List[str]) -> dict:
    """
    pdf_paths: absolute or relative paths provided by API
    """
    vector_store = get_vector_store()
    all_texts = []
    all_metadatas = []
    summary = {}

    for pdf_path in pdf_paths:
        document_name = os.path.basename(pdf_path)
        law_name = os.path.splitext(document_name)[0].replace("_", " ")

        print(f"\n📘 Processing: {document_name}")

        text = load_pdf(pdf_path)

        chunks = extract_chapters_and_sections(
            text=text,
            law_name=law_name,
            document_name=document_name
        )

        for c in chunks:
            all_texts.append(c["text"])
            all_metadatas.append(c["metadata"])

        chapter_count = Counter(c["metadata"]["chapter_number"] for c in chunks)

        summary[document_name] = {
            "chapters": len(chapter_count),
            "sections": len(chunks)
        }

        print(f"   ➤ Chapters: {len(chapter_count)}")
        print(f"   ➤ Sections: {len(chunks)}")

    if all_texts:
        vector_store.add_texts(
            texts=all_texts,
            metadatas=all_metadatas
        )

    print("✅ Upload complete")

    return summary
