
from unittest.mock import patch

from src.generation.rag_pipeline import retrieve_job_context


@patch("src.generation.rag_pipeline.semantic_search")
@patch("src.generation.rag_pipeline.load_retriever_model")
@patch("src.generation.rag_pipeline.load_collection")
def test_retrieve_job_context_success(
    mock_load_collection,
    mock_load_model,
    mock_semantic_search,
):
    mock_collection = object()
    mock_model = object()

    mock_load_collection.return_value = mock_collection
    mock_load_model.return_value = mock_model

    expected_results = [
        {
            "text": "Python and SQL are required for Data Analyst roles.",
            "metadata": {"role": "Data Analyst"},
        }
    ]

    mock_semantic_search.return_value = expected_results

    results = retrieve_job_context("Data Analyst", top_k=5)

    assert results == expected_results

    mock_load_collection.assert_called_once()
    mock_load_model.assert_called_once()

    mock_semantic_search.assert_called_once_with(
        query="What skills and qualifications are required for a Data Analyst?",
        collection=mock_collection,
        model=mock_model,
        top_k=5,
    )


@patch("src.generation.rag_pipeline.semantic_search")
@patch("src.generation.rag_pipeline.load_retriever_model")
@patch("src.generation.rag_pipeline.load_collection")
def test_retrieve_job_context_empty_results(
    mock_load_collection,
    mock_load_model,
    mock_semantic_search,
):
    mock_load_collection.return_value = object()
    mock_load_model.return_value = object()

    mock_semantic_search.return_value = []

    results = retrieve_job_context("Data Analyst", top_k=5)

    assert results == []
