from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

QUERY_CLASSIFIER_PROMPT = """
You are an expert legal query classifier for Nepali law.

Your task:
- Determine whether the question is related to law, legal rights, courts, constitution,
  statutes, acts, cases, or judicial interpretation.
- If the question is NOT related to legal or judicial matters, return NOT_LEGAL.

Classify the user's question into exactly ONE of the following labels:

- LOOKUP → asking for a specific Article or Section
- INTERPRETATION → asking meaning, scope, explanation of law
- CASE_BASED → asking how courts have interpreted or applied law
- PREDICTIVE → hypothetical or future legal outcome
- GENERAL → legal question but does not fit above categories
- NOT_LEGAL → completely unrelated to law or judiciary
- RECOMMENDATION → asking to find or recommend a lawyer,  asking to contact, consult, or book a lawyer

STRICT OUTPUT FORMAT:
- Output must be EXACTLY ONE of the labels above i.e (LOOKUP, INTERPRETATION, CASE_BASED, PREDICTIVE, GENERAL, NOT_LEGAL, RECOMMENDATION)
- Output must be in ALL CAPS
- Output must contain NO punctuation, NO spaces, NO explanation

If the question does not clearly fit any category, output GENERAL.

Question:
{question}
"""


def classify_query_llm(question: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You classify legal questions."},
            {"role": "user", "content": QUERY_CLASSIFIER_PROMPT.format(question=question)}
        ],
        temperature=0.1,
        max_tokens=10
    )
    print(f'response:{response}')

    label = response.choices[0].message.content.strip().upper()
    print(label)
    return label
