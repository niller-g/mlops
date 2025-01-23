from fastapi.testclient import TestClient
from src.mlops.api import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200}
