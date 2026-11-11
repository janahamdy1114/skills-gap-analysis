
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = "data/processed/chunks.jsonl"
BATCH_SIZE = 32


def load_chunks(path=CHUNKS_PATH):
    """Load processed chunks from a JSONL file."""

    with open(path, "r", encoding="utf-8") as f:
        chunks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not chunks:
        raise ValueError("No chunks found in the input file.")

    return chunks


def load_embedding_model(model_name=MODEL_NAME):
    """Load the Sentence Transformer embedding model."""

    model = SentenceTransformer(model_name)

    return model


def generate_embeddings(chunks, model, batch_size=BATCH_SIZE):
    """Generate an embedding vector for each chunk."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def validate_embeddings(chunks, embeddings, model):
    """Validate embedding dimensions and numerical values."""

    expected_dimension = model.get_sentence_embedding_dimension()

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of embeddings does not match number of chunks."
        )

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} dimensions, "
            f"got {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    norms = np.linalg.norm(embeddings, axis=1)

    print("\nEmbedding Validation")
    print("-" * 35)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Minimum vector norm: {norms.min():.4f}")
    print(f"Maximum vector norm: {norms.max():.4f}")
    print("All embedding values are finite.")
    print("Validation passed.")


def main():
    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print(f"Model loaded: {MODEL_NAME}")

    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(chunks, model)

    validate_embeddings(chunks, embeddings, model)


if __name__ == "__main__":
    main()
