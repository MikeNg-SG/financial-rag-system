import weaviate

from embedding import get_embedding_model

# Create hybrid search (sematic + keyword search)
# Define alpha = 0.3 (30% keyword (BM25) + 70% semantic)
def hybrid_search(client: weaviate.WeaviateClient, 
                  query: str, 
                  limit: int = 3, alpha: float = 0.3, 
                  collection_name: str ='FinancialChunk'):
    model = get_embedding_model()
    embed_vector = model.encode(query).tolist()
    collection = client.collections.get(collection_name)

    response = collection.query.hybrid(
        query=query,
        vector=embed_vector,
        alpha=alpha,
        limit=limit
    )

    return response.objects

if __name__ == "__main__":
    client = weaviate.connect_to_local()

    query = "What was Apple's revenue?"

    print("=== DENSE SEARCH (vector only) ===")
    results = hybrid_search(client, query)
    for i, obj in enumerate(results):
        props = obj.properties
        print(f"{i+1}. {props['ticker']} | {props['item']} | {props['text'][:150]}")

    print("\n=== HYBRID SEARCH (BM25 + dense) ===")
    results = hybrid_search(client, query)
    for i, obj in enumerate(results):
        props = obj.properties
        print(f"{i+1}. {props['ticker']} | {props['item']} | {props['text'][:150]}")

    client.close()