from src.generation.rag_pipeline import retrieve_job_context


def main():
    target_role = "Data Analyst"

    results = retrieve_job_context(target_role, top_k=5)

    for i, chunk in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Metadata:", chunk["metadata"])
        print("Text:", chunk["text"][:500])


if __name__ == "__main__":
    main()