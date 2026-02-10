import os
import re
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from query_classifier import classify_query_llm
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedding import embeddings
from recommendation.storage import store_user_query, fetch_user_queries
from recommendation.user_recommendation import recommend_lawyer_from_history

from prompts.prompt import (
    LOOKUP_PROMPT,
    INTERPRETATION_PROMPT,
    CASE_BASED_PROMPT,
    PREDICTIVE_PROMPT
)


SIMILARITY_THRESHOLD = 0.8


# =========================
# INITIALIZATION FUNCTION
# =========================
def init_clients():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
   

    try:
        qdrant.create_payload_index(
            collection_name="nepal_constitution",
            field_name="metadata.article_number",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass 

    qdrant.create_payload_index(
    collection_name="nepal_acts",
    field_name="metadata.section_number",
    field_schema=PayloadSchemaType.KEYWORD
)
    constitution_store = QdrantVectorStore(
        client=qdrant,
        collection_name="nepal_constitution",
        embedding=embeddings
    )
    case_store = QdrantVectorStore(
        client=qdrant,
        collection_name="case_laws",
        embedding=embeddings
    )
    act_store = QdrantVectorStore(
        client=qdrant,
        collection_name="nepal_acts",
        embedding=embeddings
    )

    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    return constitution_store, case_store, act_store, groq_client


def filter_by_similarity(results, threshold, label="DOC"):
    filtered = []
    for doc, score in results:
        
        print(f"[{label}] score={score:.4f}")

        if score <= threshold:
            

            filtered.append((doc, score))

    return filtered



# =========================
# SEARCH HELPERS
# =========================
def hybrid_constitution_search(store, query: str, k: int = 3):
    article_no = re.search(r"\barticle\s+(\d+)", query.lower()) 
    if article_no: 
        docs = store.similarity_search( 
            query, 
            k=k, 
            filter=Filter( 
                must=[FieldCondition( 
                    key="metadata.article_number", 
                    match=MatchValue(value=article_no.group(1)) 
                    )] 
                    ) 
                    ) 
        if docs: 
            return docs 
    return store.similarity_search(query, k=k)




def retrieve_act_semantic(store, query: str, k: int = 5):
    # Look for "section <number>" in the query
    section_no = re.search(r"\bsection\s+(\d+)", query.lower())

    if section_no:
        docs = store.similarity_search(
            query,
            k=k,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.section_number",
                        match=MatchValue(value=section_no.group(1))
                    )
                ]
            )
        )
        if docs:
            return docs

    # Fallback: no section number found, just do a plain similarity search
    return store.similarity_search(query, k=k)




def hybrid_case_search(store, query: str, k: int = 5):
    results = store.similarity_search_with_score(query, k=k)

    # ONLY semantic filtering — no keyword hacks
    docs = filter_by_similarity(
        results,
        SIMILARITY_THRESHOLD,
        label="CASE LAW"
    )

    return docs


# =========================
# DOCUMENT RETRIEVAL
# =========================
def retrieve_documents(question: str, query_type: str,
                       constitution_store, case_store, act_store):
    query_type = query_type.lower()
    docs = []

    if query_type == "lookup":
        if "article" in question.lower():
            docs.extend(hybrid_constitution_search(constitution_store, question))
        elif "section" in question.lower():
            docs.extend(retrieve_act_semantic(act_store, question))

    elif query_type == "case_based":
        docs.extend(hybrid_constitution_search(constitution_store, question, k=3))
        docs.extend(retrieve_act_semantic(act_store, question, k=5))
        docs.extend(hybrid_case_search(case_store, question, k=5))

    elif query_type == "predictive":
        docs.extend(hybrid_case_search(case_store, question, k=6))
        docs.extend(retrieve_act_semantic(act_store, question, k=5))

    else:
        docs.extend(hybrid_constitution_search(constitution_store, question, k=4))
        docs.extend(retrieve_act_semantic(act_store, question, k=5))

    if not docs:
        raise ValueError("No relevant legal context found.")

    return docs


# =========================
# CONTEXT FORMATTER
# =========================
def format_context(docs):
    blocks = []
    
    for item in docs:
        if isinstance(item, tuple):
            d, score = item
        else:
            d = item
            score = None
        meta = d.metadata
        doc_type = meta.get("doc_type", "").lower()

        if doc_type == "constitution":
            header = f"""[CONSTITUTION]
Article {meta.get('article_number')} – {meta.get('article_title')}
Part: {meta.get('part_title')}
Law: {meta.get('law_name')}
"""
        elif doc_type == "statute":
            header = f"""[STATUTE]
Law: {meta.get('law_name')}
Chapter {meta.get('chapter_number')} – {meta.get('chapter_title')}
Section {meta.get('section_number')} – {meta.get('section_title')}
"""
        elif doc_type == "case_law":
            header = f"""[CASE LAW]
Case: {meta.get('case_title')}
Court: {meta.get('court')}
Section: {meta.get('section')}
"""
        else:
            header = "[LEGAL DOCUMENT]"

        blocks.append(header.strip() + "\n" + d.page_content.strip())

    return "\n\n---\n\n".join(blocks)


# =========================
# PROMPT ROUTER
# =========================
def select_prompt(query_type: str) -> str:
    query_type = query_type.upper()
    if query_type == "LOOKUP":
        return LOOKUP_PROMPT
    elif query_type == "CASE_BASED":
        return CASE_BASED_PROMPT
    elif query_type == "PREDICTIVE":
        return PREDICTIVE_PROMPT
    else:
        return INTERPRETATION_PROMPT


# =========================
# GROQ CALL
# =========================
def call_groq(groq_client, prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a senior Nepali legal expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


# =========================
# MAIN RAG PIPELINE
# =========================
def legal_rag_answer(question: str, user_id: str, user_role: str):
    constitution_store, case_store, act_store, groq_client = init_clients()
    
    if user_id is None:
        user_id = str(uuid.uuid4())
    user_role = user_role.upper()

    if user_role in ["LAWYER", "FIRM"]:
        user_role = "LAWYER"
    else:
        user_role = user_role

    query_type = classify_query_llm(question)

    print(f"query: {query_type}")
    # Initialize the response dictionary
    response_data = {
        "answer": "",
        "query_type": query_type,
        "case_category": ""
    }
    
    if query_type == "RECOMMENDATION":
        # print(user_role)
        # Fetch the user's past queries from the database
        if user_role in ["LAWYER", "FIRM"]:
            response_data["answer"] = "As a lawyer, you may ask legal questions or analyze legal issues. Lawyer recommendation is available only for general users. "
                
                
            
            response_data["case_category"] = ""

        else:
        # Normal user flow
            user_queries = fetch_user_queries(user_id)
            lawyer_type = recommend_lawyer_from_history(question,user_queries)

            response_data["answer"] = ""
            response_data["case_category"] = lawyer_type
        # store_user_query(
        #     user_id=user_id,
        #     query=question,
        #     query_type=query_type,
        #     response=response_data
        # )
        
    elif query_type == "NOT_LEGAL":
        response_data["answer"] = (
            "This question is not related to Nepali law, legal provisions, "
            "or judicial matters. Please ask a legal question."
        )
        response_data["case_category"] = "" # No recommendations for non-legal queries
        
    else:
        # Process normal legal queries
        docs = retrieve_documents(question, query_type, constitution_store, case_store, act_store)
        context = format_context(docs)
        prompt_template = select_prompt(query_type)
        
        # Prepare the prompt and get response from Groq
        print('extracted prompt')
        prompt = prompt_template.format(context=context, question=question,user_role=user_role)
        # print(prompt)
        response_data["answer"] = call_groq(groq_client, prompt)
        
       
        # Store the query in the database
        # store_user_query(
        #     user_id=user_id,
        #     query=question,
        #     query_type=query_type,
        #     response=response_data
        # )
        
        response_data["case_category"] = ""  
    
    return response_data