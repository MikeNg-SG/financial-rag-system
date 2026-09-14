from sentence_transformers import CrossEncoder
import weaviate
from search import hybrid_search

def get_reranker() -> CrossEncoder:
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(reranker: CrossEncoder, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    # Build a list of (query, chunk_text) pairs — one pair per chunk
    pairs = []
    for chunk in chunks:
        pairs.append((query, chunk["text"]))

    # Run all pairs through the cross-encoder to get relevance scores
    scores = reranker.predict(pairs)

    # Attach each chunk to its score
    scored_chunks = []
    for i in range(len(chunks)):
        scored_chunks.append((chunks[i], scores[i]))

    # Sort by score, highest relevance first
    def get_score(chunk_and_score):
        return chunk_and_score[1]

    scored_chunks.sort(key=get_score, reverse=True)

    # Keep only the top_k chunks, and drop the scores
    top_chunks= []
    for chunk, score in scored_chunks[:top_k]:
        top_chunks.append(chunk)

    return top_chunks

if __name__ == "__main__":
    
    client = weaviate.connect_to_local()
    reranker = get_reranker()

    query = "What was Apple's revenue?"

    # Get a slightly larger candidate pool from hybrid search first
    results = hybrid_search(client, query, limit=10)
    candidates = [
        {"ticker": obj.properties["ticker"], "item": obj.properties["item"], "text": obj.properties["text"]}
        for obj in results
    ]

    print(" BEFORE reranking (hybrid search order)")
    for i, c in enumerate(candidates[:3]):
        print(f"{i+1}. {c['ticker']} | {c['item']} | {c['text'][:150]}")

    reranked = rerank(reranker, query, candidates, top_k=3)

    print("\n AFTER reranking (cross-encoder order)")
    for i, c in enumerate(reranked):
        print(f"{i+1}. {c['ticker']} | {c['item']} | {c['text'][:150]}")

    client.close()