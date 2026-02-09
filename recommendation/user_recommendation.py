from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LAWYER_RECOMMENDATION_PROMPT = """
You are a classification engine.

Your task:
- Analyze the user's past legal questions
- Decide which ONE lawyer category best matches the majority of the questions

Allowed categories (choose EXACTLY ONE):
- Criminal
- Civil
- Finance
- Corporate

STRICT OUTPUT RULES:
- Output MUST be exactly ONE word from the allowed categories
- Output MUST NOT contain explanations, sentences, punctuation, or extra text
- Output MUST be ONLY the category name

If the questions relate to multiple areas, choose the MOST dominant one.

User's past questions:
{queries}
"""



def recommend_lawyer_from_history(user_queries: list[str]) -> str:
    joined_queries = "\n".join(f"- {q}" for q in user_queries)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You recommend lawyer categories based on user history."},
            {"role": "user", "content": LAWYER_RECOMMENDATION_PROMPT.format(queries=joined_queries)}
        ],
        temperature=0.2,
        max_tokens=50
    )

    return response.choices[0].message.content.strip()
