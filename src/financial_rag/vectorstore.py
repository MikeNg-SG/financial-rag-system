import weaviate
from weaviate.classes.config import Configure, Property, DataType

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


if __name__ == "__main__":
    client = connect_to_weaviate()

    print("Connected to Weaviate. Is ready:", client.is_ready())

    create_collection(client)

    client.close()