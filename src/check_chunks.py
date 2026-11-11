
import json
from collections import Counter
from statistics import mean

from transformers import AutoTokenizer


CHUNKS_PATH = "data/processed/chunks.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


def load_chunks(path=CHUNKS_PATH):
    """Load chunks from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_chunk_ids(chunks):
    """Check whether all chunk IDs are unique."""
    ids = [chunk["chunk_id"] for chunk in chunks]

    print(f"Unique chunk IDs: {len(set(ids))} / {len(ids)}")
    print(f"Duplicate IDs: {len(ids) - len(set(ids))}")


def check_metadata(chunks):
    """Check required metadata fields and missing values."""
    required_fields = [
        "job_id",
        "job_title",
        "company",
        "role_category",
        "seniority_level",
        "location",
        "section",
        "source_file",
    ]

    missing_count = 0

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        for field in required_fields:
            if not metadata.get(field):
                missing_count += 1

    print(f"Missing metadata values: {missing_count}")


def check_text(chunks):
    """Check for empty chunks and inspect character lengths."""
    lengths = [len(chunk["text"]) for chunk in chunks]

    empty_count = sum(
        not chunk["text"].strip() for chunk in chunks
    )

    print(f"Empty chunks: {empty_count}")
    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk: {max(lengths)} characters")
    print(f"Average chunk length: {mean(lengths):.2f} characters")


def check_job_distribution(chunks):
    """Display the number of chunks generated for each job."""
    job_counts = Counter(
        chunk["metadata"]["job_id"] for chunk in chunks
    )

    print(f"Unique jobs represented: {len(job_counts)}")
    print(f"Minimum chunks per job: {min(job_counts.values())}")
    print(f"Maximum chunks per job: {max(job_counts.values())}")


def check_sections(chunks):
    """Display the distribution of chunks across sections."""
    sections = Counter(
        chunk["metadata"]["section"] for chunk in chunks
    )

    print("Chunks per section:", dict(sections))


def check_token_lengths(chunks):
    """Check token lengths against the embedding model's limit."""

    print("\nLoading embedding tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    token_counts = []

    for chunk in chunks:
        tokens = tokenizer.encode(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
        )

        token_counts.append({
            "chunk_id": chunk["chunk_id"],
            "count": len(tokens),
        })

    counts = [item["count"] for item in token_counts]

    exceeding = [
        item for item in token_counts
        if item["count"] > MAX_SEQ_LENGTH
    ]

    print(f"Model: {MODEL_NAME}")
    print(f"Maximum sequence length: {MAX_SEQ_LENGTH} tokens")
    print(f"Shortest chunk: {min(counts)} tokens")
    print(f"Longest chunk: {max(counts)} tokens")
    print(f"Average token length: {mean(counts):.2f} tokens")
    print(f"Chunks exceeding limit: {len(exceeding)}")

    if exceeding:
        print("\nChunks exceeding the token limit:")

        for item in exceeding:
            print(
                f"- {item['chunk_id']}: "
                f"{item['count']} tokens"
            )
    else:
        print("\nAll chunks fit within the model's token limit.")


def main():
    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        return

    print("=" * 45)
    print("CHUNK VALIDATION REPORT")
    print("=" * 45)

    print(f"\nTotal chunks: {len(chunks)}")

    check_chunk_ids(chunks)
    check_metadata(chunks)
    check_text(chunks)
    check_job_distribution(chunks)
    check_sections(chunks)
    check_token_lengths(chunks)

    print("\nValidation completed.")


if __name__ == "__main__":
    main()

