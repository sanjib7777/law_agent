# import os
# import re
# import uuid
# from typing import List, Tuple
# from docx import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_qdrant import QdrantVectorStore
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance
# from dotenv import load_dotenv
# from langchain.embeddings import Embeddings
# load_dotenv()

# # =========================
# # CONFIG
# # =========================
# COLLECTION_NAME = "case_laws"
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# CASE_SPLIT_PATTERN = r"(Case:\s*\d+)"

# SECTION_HEADERS = {
#     "facts": "brief facts",
#     "procedural_history": "procedural history",
#     "legal_principles": "legal principles",
#     "reasoning": "reasoning",
#     "decision": "final decision",
#     "summary": "case summary points"
# }

# CHUNK_CONFIG = {
#     "facts": (600, 100),
#     "procedural_history": (600, 100),
#     "legal_principles": (400, 50),
#     "reasoning": (800, 150),
#     "decision": (1000, 0),
#     "summary": (500, 50)
# }

# # Load the local model
# model = SentenceTransformer('BAAI/bge-m3')  # You can replace with your preferred model


# class LocalEmbeddings(Embeddings):
#     """LangChain-compatible embeddings using a locally loaded model"""

#     def embed_query(self, text: str):
#         return model.encode([text])[0]  # Embeds a single query

#     def embed_documents(self, texts: List[str]):
#         return model.encode(texts)  # Embeds a list of documents


# # =========================
# # LOAD DOCX
# # =========================
# def load_docx(path: str) -> str:
#     doc = Document(path)
#     return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# # =========================
# # SPLIT CASES
# # =========================
# def split_cases(text: str) -> List[str]:
#     parts = re.split(CASE_SPLIT_PATTERN, text)
#     cases, buffer = [], ""

#     for part in parts:
#         if re.match(r"Case:\s*\d+", part.strip()):
#             if buffer.strip():
#                 cases.append(buffer.strip())
#             buffer = part
#         else:
#             buffer += "\n" + part

#     if buffer.strip():
#         cases.append(buffer.strip())

#     return cases


# # =========================
# # METADATA EXTRACTION
# # =========================
# def extract_case_metadata(text: str) -> dict:
#     def find(pattern):
#         m = re.search(pattern, text, re.IGNORECASE)
#         return m.group(1).strip() if m and m.groups() else ""

#     return {
#         "doc_type": "case_law",
#         "case_index": find(r"Case:\s*(\d+)"),
#         "case_title": find(r"Case Title\s*:\s*(.+)"),
#         "court": find(r"(Supreme Court[^\n]*)"),
#         "decision_date": find(r"Decision Date\s*:\s*(.+)"),
#         "case_no": find(r"Case No\s*:\s*(.+)"),
#         "subject": find(r"Subject\s*:\s*(.+)")
#     }


# # =========================
# # SECTION EXTRACTION
# # =========================
# def extract_sections(case_text: str) -> dict:
#     sections = {}
#     current, buffer = None, []

#     for line in case_text.splitlines():
#         line_clean = line.strip()
#         lower = line_clean.lower()

#         matched = False
#         for key, header in SECTION_HEADERS.items():
#             if header in lower:
#                 if current:
#                     sections[current] = "\n".join(buffer).strip()
#                 current = key
#                 buffer = []
#                 matched = True
#                 break

#         if not matched and current:
#             buffer.append(line_clean)

#     if current:
#         sections[current] = "\n".join(buffer).strip()

#     return sections


# # =========================
# # CHUNKING
# # =========================
# def chunk_sections(sections: dict, base_metadata: dict) -> Tuple[list, list]:
#     texts, metadatas = [], []

#     for section, content in sections.items():
#         if not content.strip():
#             continue

#         chunk_size, overlap = CHUNK_CONFIG.get(section, (600, 100))
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=overlap
#         )

#         for idx, chunk in enumerate(splitter.split_text(content)):
#             texts.append(chunk)
#             metadatas.append({
#                 **base_metadata,
#                 "section": section,
#                 "chunk_index": idx,
#                 "doc_id": str(uuid.uuid4()),
#                 "source": "case_law"
#             })

#     return texts, metadatas


# # =========================
# # VECTOR STORE
# # =========================
# def get_vector_store() -> QdrantVectorStore:
#     client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

#     if not client.collection_exists(COLLECTION_NAME):
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=1024,
#                 distance=Distance.COSINE
#             )
#         )

#     return QdrantVectorStore(
#         client=client,
#         collection_name=COLLECTION_NAME,
#         embedding=LocalEmbeddings() 
#     )


# # =========================
# # INGEST ENTRY POINT
# # =========================
# def ingest_case_docx(docx_path: str) -> dict:
#     text = load_docx(docx_path)
#     cases = split_cases(text)
#     print(len(cases))
#     vector_store = get_vector_store()
#     total_chunks = 0

#     for case_text in cases:
#         metadata = extract_case_metadata(case_text)
#         sections = extract_sections(case_text)
#         texts, metas = chunk_sections(sections, metadata)

#         if texts:
#             vector_store.add_texts(texts, metadatas=metas)
#             total_chunks += len(texts)
#     print(total_chunks)
#     return {
#         "status": "success",
#         "cases_detected": len(cases),
#         "chunks_uploaded": total_chunks,
#         "collection": COLLECTION_NAME
#     }

# # =========================
# # RUN THE INGESTION FOR INDIVIDUAL FILES
# # =========================
# if __name__ == "__main__":
#     docx_paths = [
#         "./dataset/criminal-kartabya jyan.docx",
#         "./dataset/Personal-Criminal.docx"
#     ]
#     for docx_path in docx_paths:
#         print(f"Ingesting file: {docx_path}")
#         result = ingest_case_docx(docx_path)
#         print(result)