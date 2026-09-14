import weaviate
from weaviate.classes.config import Configure, Property, DataType
from pathlib import Path
import json

from financial_rag.embedding import get_embedding_model

# Connect to the local Weaviate via Docker
def connect_to_weaviate() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local()

# Create a Weaviate collection to store the chunks
def create_collection(client: weaviate.WeaviateClient, collection_name: str = "FinancialChunk"):
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists — deleting and recreating")
        client.collections.delete(collection_name)

    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="ticker", data_type=DataType.TEXT),
            Property(name="item", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),  # Use the local embedding model
    )
    print(f"Created collection '{collection_name}'")

def load_all_chunks(companies: list[str], input_dir: str = 'data/processed') -> list[dict]:
    all_chunks = []
    in_path = Path(input_dir)
    for ticker in companies:
        chunks_file = in_path / f"{ticker}_chunks.json"
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    return all_chunks

def embed_and_insert(client: weaviate.WeaviateClient, chunks: list[dict], collection_name: str = "FinancialChunk"):
    model = get_embedding_model()
    collection = client.collections.get(collection_name)

    print(f"Embedding and inserting {len(chunks)} chunks...")

    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk["text"]).tolist()

            batch.add_object(
                properties={
                    "ticker": chunk["ticker"],
                    "item": chunk["item"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
                vector=vector,
            )

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(chunks)} done")

    print("Done inserting.")
    

if __name__ == "__main__":
    client = connect_to_weaviate()
    print("Connected to Weaviate. Is ready:", client.is_ready())

    create_collection(client)

    companies = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    chunks = load_all_chunks(companies)
    print(f"Loaded {len(chunks)} total chunks from disk")

    embed_and_insert(client, chunks)

    # Quick sanity check: count how many objects actually got stored
    collection = client.collections.get("FinancialChunk")
    count = collection.aggregate.over_all(total_count=True)
    print(f"\nTotal objects in Weaviate: {count.total_count}")

    client.close()