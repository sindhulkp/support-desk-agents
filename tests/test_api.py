"""
tests/test_api.py — the FastAPI endpoints, via the test client.

No server is started: TestClient calls the app in-process. The API runs
in mock mode here (no API key in the test environment), which is exactly
what the demo and CI use.
"""

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_returns_ok_and_mock_mode():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "mock"
    assert body["tickets_processed"] == 0


def test_homepage_serves_the_live_demo_page():
    """GET / serves the demo frontend (the page anyone opens), not JSON."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Support Desk Agents" in response.text
    assert "Run the agents" in response.text
    # The page must not depend on any outside service or branding.
    assert "muse.ai" not in response.text


def test_post_sample_ticket_and_get_stored_result():
    post = client.post("/tickets", json={"ticket_id": "T-1002"})
    assert post.status_code == 201
    body = post.json()
    assert body["classification"]["category"] == "technical"
    assert body["retrieved"][0]["title"] == "Password Reset"
    assert "Password Reset" in body["reply"]

    stored = client.get("/tickets/T-1002")
    assert stored.status_code == 200
    assert stored.json()["ticket"]["id"] == "T-1002"


def test_post_inline_ticket():
    post = client.post(
        "/tickets",
        json={
            "id": "WEB-1",
            "customer_id": "C-200",
            "customer_name": "Sam",
            "subject": "Where is my refund?",
            "message": "I returned my order two weeks ago and still have no refund.",
        },
    )
    assert post.status_code == 201
    assert post.json()["classification"]["category"] == "billing"


def test_unknown_ticket_ids_return_404():
    assert client.post("/tickets", json={"ticket_id": "T-9999"}).status_code == 404
    assert client.get("/tickets/T-9999").status_code == 404


def test_post_with_no_ticket_and_no_text_returns_422():
    assert client.post("/tickets", json={}).status_code == 422
