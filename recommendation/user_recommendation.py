from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LAWYER_RECOMMENDATION_PROMPT = """
You are a legal assistant for Nepal.

Below is a list of a user's past legal questions.
Analyze them and determine which type of lawyer the user most likely needs.

Possible categories:
- Criminal 
- Civil 
- Finance
- Corporate 

Return ONLY ONE label from:
Criminal, Civil, Finance, Corporate

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
