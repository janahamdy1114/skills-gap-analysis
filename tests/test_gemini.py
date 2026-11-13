
from unittest.mock import patch, MagicMock

import pytest

from src.generation.llm_client import generate_response


def test_generate_response_success():
    mock_response = MagicMock()
    mock_response.text = "Machine Learning is a field of AI."

    with patch(
        "src.generation.llm_client.client.models.generate_content",
        return_value=mock_response,
    ) as mock_generate:

        answer = generate_response("Explain Machine Learning")

        assert answer == "Machine Learning is a field of AI."
        mock_generate.assert_called_once()


def test_generate_response_empty_prompt():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generate_response("")


def test_generate_response_whitespace_prompt():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generate_response("   ")
