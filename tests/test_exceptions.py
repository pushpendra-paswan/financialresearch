from fastapi.testclient import TestClient

from app.exceptions import ConflictError, ForbiddenError, NotFoundError, UnauthorizedError
from app.main import app


# Throwaway routes that exist only for this test module, so the exception handlers in
# main.py are tested without needing a real feature
@app.get("/test-only/not-found")
def raise_not_found() -> None:
    raise NotFoundError("Thing not found")


@app.get("/test-only/conflict")
def raise_conflict() -> None:
    raise ConflictError("Thing already exists")


@app.get("/test-only/unauthorized")
def raise_unauthorized() -> None:
    raise UnauthorizedError("Not authenticated")


@app.get("/test-only/forbidden")
def raise_forbidden() -> None:
    raise ForbiddenError("Not allowed")


def test_not_found_error_returns_404(client: TestClient) -> None:
    response = client.get("/test-only/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "Thing not found"}


def test_conflict_error_returns_409(client: TestClient) -> None:
    response = client.get("/test-only/conflict")

    assert response.status_code == 409
    assert response.json() == {"detail": "Thing already exists"}


def test_unauthorized_error_returns_401_with_www_authenticate_header(client: TestClient) -> None:
    response = client.get("/test-only/unauthorized")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_forbidden_error_returns_403(client: TestClient) -> None:
    response = client.get("/test-only/forbidden")

    assert response.status_code == 403
    assert response.json() == {"detail": "Not allowed"}
