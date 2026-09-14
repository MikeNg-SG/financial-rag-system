import weaviate

from embedding import get_embedding_model


def search(client: weaviate.WeaviateClient, query: str, limit: int = 3, collection_name: str = "FinancialChunk"):
    """Embed a query, search Weaviate, return the most similar chunks."""
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    collection = client.collections.get(collection_name)

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
    )

    return response.objects


if __name__ == "__main__":
    client = weaviate.connect_to_local()

    query = "What was Apple's revenue?"
    results = search(client, query)

    print(f"Query: {query}\n")
    for i, obj in enumerate(results):
        props = obj.properties
        print(f"--- Result {i+1}: {props['ticker']} | {props['item']} ---")
        print(props["text"][:300])
        print()

    client.close()