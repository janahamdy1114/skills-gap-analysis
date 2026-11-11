from src.embeddings.embedding_model import load_embedding_model
from src.retrieval.vector_store import (
    create_chroma_client,
    COLLECTION_NAME,
    DB_PATH,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_collection():
    """Connect to the existing persistent ChromaDB collection."""
    client = create_chroma_client(DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. " "Run vector_store.py first."
        ) from exc

    return collection


def load_retriever_model():
    """Load the same embedding model used to index the documents."""
    return load_embedding_model(MODEL_NAME)


def embed_query(query, model):
    """Generate a normalized embedding for a user query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.tolist()


def semantic_search(query, collection, model, top_k=5, where=None):
    """
    Retrieve the top-k most relevant chunks.

    Args:
        query: Natural-language search query.
        collection: ChromaDB collection.
        model: Loaded SentenceTransformer model.
        top_k: Maximum number of results.
        where: Optional ChromaDB metadata filter.

    Returns:
        A list of dictionaries containing retrieved chunks,
        metadata, distances, and similarity scores.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query, model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]

        retrieved_chunks.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": 1 - distance,
            }
        )

    return retrieved_chunks


def get_chunks_by_role(
    collection,
    role_category=None,
    job_title=None,
    seniority_level=None,
):
    """
    Retrieve all chunks matching the supplied metadata filters.

    This is useful for inspecting the skills and requirements
    associated with a particular job role.
    """
    filters = []

    if role_category:
        filters.append({"role_category": role_category})

    if job_title:
        filters.append({"job_title": job_title})

    if seniority_level:
        filters.append({"seniority_level": seniority_level})

    if not filters:
        raise ValueError("At least one metadata filter is required.")

    where = filters[0] if len(filters) == 1 else {"$and": filters}

    results = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    retrieved_chunks = []

    for i in range(len(results["ids"])):
        retrieved_chunks.append(
            {
                "chunk_id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            }
        )

    return retrieved_chunks


def display_results(results):
    """Print retrieved chunks in a readable format."""
    if not results:
        print("No matching chunks found.")
        return

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n{'=' * 70}")
        print(f"Result {i}")
        print(f"{'=' * 70}")

        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Job Title: {metadata['job_title']}")
        print(f"Company: {metadata['company']}")
        print(f"Role Category: {metadata['role_category']}")
        print(f"Seniority: {metadata['seniority_level']}")

        if "similarity" in result:
            print(f"Cosine Distance: {result['distance']:.4f}")
            print(f"Similarity Score: {result['similarity']:.4f}")

        print(f"\nText:\n{result['text']}")


if __name__ == "__main__":
    collection = load_collection()
    model = load_retriever_model()

    print(f"Collection contains {collection.count()} chunks.")

    query = "What skills are required for machine learning engineers?"

    results = semantic_search(
        query=query,
        collection=collection,
        model=model,
        top_k=5,
    )

    print("\nSEMANTIC SEARCH RESULTS")
    display_results(results)
    print("\nMETADATA-FILTERED RETRIEVAL RESULTS")

    role_results = get_chunks_by_role(
      collection=collection,
      role_category="Machine Learning",
    )

    print(f"Matching chunks: {len(role_results)}")
    display_results(role_results[:5])

    print("\nCOMBINED SEMANTIC + METADATA RETRIEVAL")

    filtered_results = semantic_search(
    query="What programming languages and frameworks are required?",
    collection=collection,
    model=model,
    top_k=5,
    where={
        "role_category": "Machine Learning"
     },
    )

    display_results(filtered_results)