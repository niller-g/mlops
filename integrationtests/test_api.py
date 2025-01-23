# tests/test_api.py

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI "app" from your code
from src.mlops.api import app


# Create a mock model and tokenizer for testing
@pytest.fixture(autouse=True)
def mock_model_setup():
    with (
        patch("mlops.model.DistilGPT2Model.from_pretrained") as mock_model,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
    ):
        # Configure the mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        yield


@pytest.fixture
def client():
    # This ensures the lifespan context is properly handled
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """
    Basic test to ensure the root endpoint works
    and returns the expected JSON payload.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


@patch("mlops.predict.generate_text", return_value="[MOCKED] generated text")
def test_infer_mocked(mock_generate_text, client):
    """
    Test the /infer/{prompt} endpoint with a mocked generate_text function,
    so we don't need a real model or local files.
    """
    prompt = "test_prompt"
    max_length = 16
    response = client.post(f"/infer/{prompt}?max_length={max_length}")
    assert response.status_code == 200
    assert response.text == '"[MOCKED] generated text"'
    mock_generate_text.assert_called_once()
    assert len(response.text) > 0


@pytest.mark.skipif(
    not os.path.isdir("models/distilgpt2-finetuned-final"),
    reason="Local model folder not found. Skipping real integration test.",
)
def test_infer_real(client):
    """
    Integration test: actually load the local model if it exists,
    then hit the /infer endpoint for a real inference.
    """
    prompt = "real_model_test"
    max_length = 16
    response = client.post(f"/infer/{prompt}?max_length={max_length}")
    assert response.status_code == 200
    assert len(response.text) > 0
    print("Real model response:", response.text)
