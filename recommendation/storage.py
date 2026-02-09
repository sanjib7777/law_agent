import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

# Read the full connection string from .env
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )

def store_user_query(
    user_id: str,
    query: str,
    query_type: str,
    response: str 
):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO ai_assistant_aiqueryhistory
        (user_id, query, query_type,response)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, query, query_type, Json(response))
    )

    conn.commit()
    cur.close()
    conn.close()

def fetch_user_queries(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT query
        FROM ai_assistant_aiqueryhistory
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [r["query"] for r in rows]