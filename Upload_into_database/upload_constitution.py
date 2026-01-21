import fitz 
import re
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from embedding import embeddings
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nepal_constitution"
MAX_CONSTITUTION_PAGE = 220


# ==================
# TOC (STATIC & TRUSTED)
# =========================
toc = [
    {"title": "Preamble", "page": 6},
    {"title": "Part 1: Preliminary", "page": 8},
    {"title": "Part 2: Citizenship", "page": 10},
    {"title": "Part 3: Fundamental Rights and Duties", "page": 13},
    {"title": "Part 4: Directive Principles, Policies and Obligations of the State", "page": 30},
    {"title": "Part 5: Structure of State and Distribution of State Power", "page": 45},
    {"title": "Part 6: President and Vice-President", "page": 50},
    {"title": "Part 7: Federal Executive", "page": 54},
    {"title": "Part 8: Federal Legislature", "page": 59},
    {"title": "Part 9: Federal Legislative Procedures", "page": 75},
    {"title": "Part 10: Federal Financial Procedures", "page": 80},
    {"title": "Part 11: Judiciary", "page": 85},
    {"title": "Part 12: Attorney General", "page": 104},
    {"title": "Part 13: State Executive", "page": 108},
    {"title": "Part 14: State Legislature", "page": 115},
    {"title": "Part 15: State Legislative Procedures", "page": 126},
    {"title": "Part 16: State Financial Procedures", "page": 130},
    {"title": "Part 17: Local Executive", "page": 134},
    {"title": "Part 18: Local Legislature", "page": 140},
    {"title": "Part 19: Local Financial Procedures", "page": 143},
    {"title": "Part 20: Interrelations between Federation, State and Local Level", "page": 144},
    {"title": "Part 21: CIAA", "page": 148},
    {"title": "Part 22: Auditor General", "page": 152},
    {"title": "Part 23: Public Service Commission", "page": 155},
    {"title": "Part 24: Election Commission", "page": 160},
    {"title": "Part 25: National Human Rights Commission", "page": 163},
    {"title": "Part 26: National Natural Resources and Fiscal Commission", "page": 168},
    {"title": "Part 27: Other Commissions", "page": 172},
    {"title": "Part 28: National Security", "page": 185},
    {"title": "Part 29: Political Parties", "page": 188},
    {"title": "Part 30: Emergency Power", "page": 191},
    {"title": "Part 31: Amendment", "page": 194},
    {"title": "Part 32: Miscellaneous", "page": 196},
    {"title": "Part 33: Transitional Provisions", "page": 207},
    {"title": "Part 34: Definitions", "page": 218},
    {"title": "Part 35: Short Title", "page": 220},
]

# =========================
# ARTICLE SPLITTER
# =========================
def split_into_articles(text: str) -> List[Dict]:
    pattern = r'^\s*(\d+)\.\s+(.*)$'
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))

    articles = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        title_line = m.group(2).split(":", 1)[0].strip()

        articles.append({
            "article_number": m.group(1),
            "article_title": title_line,
            "article_text": text[start:end].strip()
        })

    return articles


# =========================
# EXTRACT ARTICLES
# =========================
def extract_constitution_articles(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    all_articles = []

    for i, part in enumerate(toc):
        start_page = part["page"] - 1
        if start_page + 1 > MAX_CONSTITUTION_PAGE:
            break

        raw_end = toc[i + 1]["page"] - 1 if i + 1 < len(toc) else doc.page_count
        end_page = min(raw_end, MAX_CONSTITUTION_PAGE)

        part_text = ""
        for p in range(start_page, end_page):
            part_text += doc[p].get_text()

        articles = split_into_articles(part_text)
        print(len(articles))

        for art in articles:
            all_articles.append({
                "text": art["article_text"],
                "metadata": {
                    "doc_type": "constitution",
                    "law_name": "Constitution of Nepal 2015",
                    "part_title": part["title"],
                    "article_number": art["article_number"],
                    "article_title": art["article_title"],
                    "start_page": start_page + 1,
                    "end_page": end_page
                }
            })

    return all_articles


# =========================
# INGEST TO QDRANT
# =========================
def ingest_constitution(pdf_path: str):
    articles = extract_constitution_articles(pdf_path)

    print("article extracted....",len(articles))
    

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,timeout=60)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embeddings.client.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
    print("collection created...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    vector_store.add_texts(
        texts=[a["text"] for a in articles],
        metadatas=[a["metadata"] for a in articles]
    )

    return {
        "status": "success",
        "articles_ingested": len(articles),
        "collection": COLLECTION_NAME
    }
