
import chromadb

from src.embeddings.embedding_model import (
    load_chunks,
    load_embedding_model,
    generate_embeddings,
)


# Configuration
CHUNKS_PATH = "data/processed/chunks.jsonl"
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "skills_gap_jobs"
BATCH_SIZE = 32


def create_chroma_client(path=DB_PATH):
    """Create a persistent ChromaDB client."""

    client = chromadb.PersistentClient(path=path)

    return client


def get_or_create_collection(client, name=COLLECTION_NAME):
    """Get an existing collection or create a new one."""

    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "description": "AI and ML job description embeddings",
            "hnsw:space": "cosine",
        },
    )

    return collection


def store_chunks(collection, chunks, embeddings):
    """Store chunk IDs, text, metadata, and embeddings in ChromaDB."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks does not match number of embeddings."
        )

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return len(ids)


def validate_collection(collection, expected_count):
    """Check whether all expected records exist in the collection."""

    actual_count = collection.count()

    print("\nChromaDB Validation")
    print("-" * 35)
    print(f"Collection name: {collection.name}")
    print(f"Expected records: {expected_count}")
    print(f"Stored records: {actual_count}")

    if actual_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} records, "
            f"but found {actual_count}."
        )

    print("All records stored successfully.")

    # Inspect one stored record
    sample = collection.peek(limit=1)

    print("\nExample stored record:")
    print(f"ID: {sample['ids'][0]}")
    print(f"Document: {sample['documents'][0][:150]}...")
    print(f"Metadata: {sample['metadatas'][0]}")

    print("\nValidation passed.")


def build_vector_store():
    """Build the complete embedding and vector storage pipeline."""

    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(
        chunks,
        model,
        batch_size=BATCH_SIZE,
    )

    print(f"Generated embeddings: {embeddings.shape}")

    print("\nConnecting to ChromaDB...")
    client = create_chroma_client()

    collection = get_or_create_collection(client)

    print(f"Collection ready: {collection.name}")

    print("\nStoring chunks...")
    stored_count = store_chunks(
        collection,
        chunks,
        embeddings,
    )

    print(f"Records submitted: {stored_count}")

    validate_collection(collection, len(chunks))


if __name__ == "__main__":
    build_vector_store()
