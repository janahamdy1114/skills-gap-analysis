import json
from pathlib import Path

from src.retrieval.retriever import (
    load_collection,
    load_retriever_model,
    semantic_search,
)

QUERIES_PATH = "data/processed/evaluation_queries.json"

K_VALUES = [5, 10]

# Retrieve a larger candidate pool because multiple chunks
# may belong to the same job description.
CANDIDATE_K = 316


def load_evaluation_queries(path=QUERIES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_unique_jobs(results):
    """Keep the first (highest-ranked) chunk from each job."""
    unique_results = []
    seen_jobs = set()

    for result in results:
        job_id = result["metadata"]["job_id"]

        if job_id not in seen_jobs:
            seen_jobs.add(job_id)
            unique_results.append(result)

    return unique_results


def calculate_metrics(ranked_job_ids, relevant_job_ids, k):
    relevant = set(relevant_job_ids)
    retrieved = ranked_job_ids[:k]

    relevant_retrieved = sum(1 for job_id in retrieved if job_id in relevant)

    denominator = min(k, len(ranked_job_ids))

    precision = relevant_retrieved / denominator if denominator > 0 else 0.0

    recall = relevant_retrieved / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0

    for rank, job_id in enumerate(ranked_job_ids, start=1):
        if job_id in relevant:
            reciprocal_rank = 1 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
    }


def display_query_results(result, relevant_job_ids):
    """Display retrieval performance and the top 10 retrieved jobs."""

    relevant = set(relevant_job_ids)
    ranked_jobs = result["retrieved_jobs"]

    print("\n" + "-" * 65)
    print("PER-QUERY RESULTS")
    print("-" * 65)

    print(f"Query: {result['query']}")
    print(f"Total relevant jobs: {len(relevant)}")
    print(f"Unique jobs retrieved: {len(ranked_jobs)}")

    # Display metrics
    print("\nMetrics:")

    for k, values in result["metrics"].items():
        print(
            f"@{k}: "
            f"Precision = {values['precision']:.4f}, "
            f"Recall = {values['recall']:.4f}, "
            f"MRR = {values['mrr']:.4f}"
        )

    # Find the first relevant result
    first_relevant_rank = next(
        (
            rank
            for rank, job_id in enumerate(ranked_jobs, start=1)
            if job_id in relevant
        ),
        None,
    )

    print(
        f"\nFirst relevant job rank: "
        f"{first_relevant_rank if first_relevant_rank else 'Not found'}"
    )

    # Show the top 10 retrieved jobs
    print("\nTop 10 retrieved jobs:")

    for rank, job_id in enumerate(ranked_jobs[:10], start=1):
        status = "RELEVANT" if job_id in relevant else "NOT RELEVANT"
        print(f"{rank:2}. {job_id:<12} {status}")

    # Show relevant jobs that weren't retrieved in the top 10
    retrieved_top_10 = set(ranked_jobs[:10])
    missed_jobs = relevant - retrieved_top_10

    print("\nRelevant jobs not retrieved in the top 10:")

    if missed_jobs:
        print(", ".join(sorted(missed_jobs)))
    else:
        print("None")


def evaluate_query(query_data, collection, model):
    query = query_data["query"]
    relevant_job_ids = query_data["relevant_job_ids"]
    where = query_data.get("where")

    if not relevant_job_ids:
        print(f"Skipping unlabeled query: {query}")
        return None

    results = semantic_search(
        query=query,
        collection=collection,
        model=model,
        top_k=CANDIDATE_K,
        where=where,
    )

    unique_results = get_unique_jobs(results)

    ranked_job_ids = [result["metadata"]["job_id"] for result in unique_results]

    metrics = {}

    for k in K_VALUES:
        metrics[k] = calculate_metrics(
            ranked_job_ids,
            relevant_job_ids,
            k,
        )

    first_relevant_rank = next(
        (
            rank
            for rank, job_id in enumerate(ranked_job_ids, start=1)
            if job_id in set(relevant_job_ids)
        ),
        None,
    )

    return {
        "query": query,
        "retrieved_jobs": ranked_job_ids,
        "relevant_job_ids": relevant_job_ids,
        "first_relevant_rank": first_relevant_rank,
        "metrics": metrics,
    }


def run_evaluation():
    queries = load_evaluation_queries()

    collection = load_collection()
    model = load_retriever_model()

    print(f"Collection size: {collection.count()}")
    print(f"Total evaluation queries: {len(queries)}")

    all_results = []

    for index, query_data in enumerate(queries, start=1):
        print(f"\nEvaluating query {index}:")
        print(query_data["query"])

        result = evaluate_query(
            query_data,
            collection,
            model,
        )
        if result is not None:
            all_results.append(result)
            display_query_results(result, query_data["relevant_job_ids"])

    if not all_results:
        raise ValueError(
            "No labeled queries found. " "Fill in relevant_job_ids before evaluation."
        )

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    for k in K_VALUES:
        avg_precision = sum(
            result["metrics"][k]["precision"] for result in all_results
        ) / len(all_results)

        avg_recall = sum(
            result["metrics"][k]["recall"] for result in all_results
        ) / len(all_results)

        avg_mrr = sum(result["metrics"][k]["mrr"] for result in all_results) / len(
            all_results
        )

        print(f"\nMetrics at K = {k}")
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Average Recall@{k}:    {avg_recall:.4f}")
        print(f"Average MRR:           {avg_mrr:.4f}")

    print(f"\nEvaluated queries: {len(all_results)}")


if __name__ == "__main__":
    run_evaluation()
