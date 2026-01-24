from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv
load_dotenv()
def print_unique_statute_laws():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    unique_laws = set()
    offset = None
    limit = 100  # batch size

    while True:
        points, offset = client.scroll(
            collection_name="nepal_acts",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_type",
                        match=MatchValue(value="statute")
                    )
                ]
            ),
            with_payload=True,
            limit=limit,
            offset=offset
        )

        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            law_name = meta.get("law_name")

            if law_name:
                unique_laws.add(law_name)

        if offset is None:
            break

    print("📜 Unique Statute Laws in nepal_acts:\n")
    for law in sorted(unique_laws):
        print("-", law)

print_unique_statute_laws()