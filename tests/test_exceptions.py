from fastapi.testclient import TestClient

from app.exceptions import ConflictError, NotFoundError
from app.main import app


# Throwaway routes that exist only for this test module, so the exception handlers in
# main.py are tested without needing a real feature
@app.get("/test-only/not-found")
def raise_not_found() -> None:
    raise NotFoundError("Thing not found")


@app.get("/test-only/conflict")
def raise_conflict() -> None:
    raise ConflictError("Thing already exists")


def test_not_found_error_returns_404(client: TestClient) -> None:
    response = client.get("/test-only/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "Thing not found"}


def test_conflict_error_returns_409(client: TestClient) -> None:
    response = client.get("/test-only/conflict")

    assert response.status_code == 409
    assert response.json() == {"detail": "Thing already exists"}
