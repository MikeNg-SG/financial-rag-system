from sentence_transformers import CrossEncoder
import weaviate
from financial_rag.search import hybrid_search

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

def retrieve(client, reranker, query: str, top_k: int = 3, candidate_pool: int = 10) -> list[dict]:
    # Load the cross-encoder model
    reranker = get_reranker()
    # Get a wide pool of candidate from hybrid search
    results = hybrid_search(client, query, limit=candidate_pool)\
    # Weaviate returns its own object type — convert each result into a
    # plain dict so rerank() can work with it the same way it works with
    # chunks anywhere else in the project 
    candidates = []
    for obj in results:
        new_dict = {
            "ticker": obj.properties["ticker"],
            "item": obj.properties["item"],
            "text": obj.properties["text"],
        }
        candidates.append(new_dict)

    return rerank(reranker, query, candidates, top_k=top_k)

if __name__ == "__main__":
    
    client = weaviate.connect_to_local()
    reranker = get_reranker()  # load once, reuse across calls

    query = "What was Apple's revenue?"
    top_chunks = retrieve(client, reranker, query)

    print(f"Query: {query}\n")
    for i, chunk in enumerate(top_chunks):
        print(f"{i+1}. {chunk['ticker']} | {chunk['item']}")
        print(f"   {chunk['text'][:200]}")
        print()

    client.close()
