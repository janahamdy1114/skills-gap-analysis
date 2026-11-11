
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingest import load_csv_jobs


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text):
    """Normalize whitespace and remove unnecessary spacing."""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def build_chunks(jobs):
    """Split job descriptions into chunks while preserving job metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for job in jobs:
        job_id = job["job_id"]

        # Clean the job description
        description = clean_text(job["description"])

        if not description:
            continue

        # Add context to help the embedding model identify the job
        full_text = (
            f"{job['job_title']} at {job['company']}: "
            f"{description}"
        )

        # Split the description into smaller pieces
        pieces = splitter.split_text(full_text)

        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{job_id}_full_description_{i}",
                "text": piece,
                "metadata": {
                    "job_id": job_id,
                    "job_title": job["job_title"],
                    "company": job["company"],
                    "role_category": job["role_category"],
                    "seniority_level": job["seniority_level"],
                    "location": job["location"],
                    "section": "full_description",
                    "source_file": job["source_file"],
                },
            })

    return chunks


def save_chunks(chunks, out_path="data/processed/chunks.jsonl"):
    """Save chunks as JSON Lines."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    jobs = load_csv_jobs()
    chunks = build_chunks(jobs)

    save_chunks(chunks)

    print(f"Jobs loaded: {len(jobs)}")
    print(f"Chunks created: {len(chunks)}")
    print("Saved to data/processed/chunks.jsonl")

    print("\nExample chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
