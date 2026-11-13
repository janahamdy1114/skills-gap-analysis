
from unittest.mock import patch

from src.generation.rag_pipeline import (
    analyze_skills_gap,
    build_analysis_prompt,
)


def test_build_analysis_prompt():
    retrieved_chunks = [
        {"text": "Python and SQL are required skills."},
        {"text": "Data visualization is also important."},
    ]

    prompt = build_analysis_prompt(
        user_skills=["Python", "SQL"],
        target_role="Data Analyst",
        retrieved_chunks=retrieved_chunks,
    )

    assert "Data Analyst" in prompt
    assert "Python, SQL" in prompt
    assert "Python and SQL are required skills." in prompt
    assert "Data visualization is also important." in prompt


@patch("src.generation.rag_pipeline.generate_response")
@patch("src.generation.rag_pipeline.retrieve_job_context")
def test_analyze_skills_gap_success(mock_retrieve, mock_generate):
    mock_retrieve.return_value = [
        {"text": "Python and SQL are required skills."}
    ]

    mock_generate.return_value = "Your skills gap analysis."

    result = analyze_skills_gap(
        user_skills=["Python", "SQL"],
        target_role="Data Analyst",
        top_k=5,
    )

    assert result == "Your skills gap analysis."

    mock_retrieve.assert_called_once_with(
        target_role="Data Analyst",
        top_k=5,
    )

    mock_generate.assert_called_once()


@patch("src.generation.rag_pipeline.generate_response")
@patch("src.generation.rag_pipeline.retrieve_job_context")
def test_analyze_skills_gap_empty_context(mock_retrieve, mock_generate):
    mock_retrieve.return_value = []

    mock_generate.return_value = "Retrieved context is insufficient."

    result = analyze_skills_gap(
        user_skills=["Python"],
        target_role="Data Analyst",
        top_k=5,
    )

    assert result == "Retrieved context is insufficient."
    mock_generate.assert_called_once()
