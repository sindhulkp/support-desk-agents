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


def test_stats_counts_processed_tickets_and_escalations():
    """GET /stats aggregates the stored results: totals, breakdowns,
    escalations, honest 'I don't know' answers, and recent tickets."""
    client.post("/tickets", json={"ticket_id": "T-1001"})
    client.post("/tickets", json={"ticket_id": "T-1009"})

    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["tickets_processed"] == 2
    assert body["by_category"]["billing"] == 2
    assert body["escalated"] == 1
    assert body["said_i_dont_know"] == 1
    assert body["customers_seen"] == 2
    assert [t["id"] for t in body["recent"]] == ["T-1009", "T-1001"]


def test_post_inline_ticket_with_order_number_returns_order():
    """A ticket asking about a real order number gets that order in
    the response, and the reply states its real status."""
    post = client.post(
        "/tickets",
        json={
            "subject": "Where is my order #1001?",
            "message": "Hi, I ordered last week, where is my order #1001?",
        },
    )
    assert post.status_code == 201
    body = post.json()
    assert body["order_id_mentioned"] == "ORD-1001"
    assert body["order"]["order_id"] == "ORD-1001"
    assert body["order"]["status"] == "Shipped"
    assert "shipped" in body["reply"].lower()


def test_post_inline_ticket_with_unknown_order_is_honest():
    """An order number that does not exist is reported honestly —
    order null, no invented status in the reply."""
    post = client.post(
        "/tickets",
        json={
            "subject": "Where is my order #9999?",
            "message": "Hi, where is my order #9999? I placed it last week.",
        },
    )
    assert post.status_code == 201
    body = post.json()
    assert body["order_id_mentioned"] == "ORD-9999"
    assert body["order"] is None
    assert "couldn't find" in body["reply"].lower()


def test_post_with_email_identifies_customer_and_second_is_repeat():
    """An email becomes the customer id (lowercased), so the same
    person writing twice is recognised as a repeat customer."""
    first = client.post(
        "/tickets",
        json={
            "customer_name": "Jordan Lee",
            "customer_email": "Jordan@example.com",
            "subject": "Charged twice",
            "message": "I was charged twice for my subscription, please refund me.",
        },
    )
    assert first.status_code == 201
    assert first.json()["ticket"]["customer_id"] == "jordan@example.com"
    assert first.json()["prior_tickets"] == 0

    second = client.post(
        "/tickets",
        json={
            "customer_name": "Jordan Lee",
            "customer_email": "jordan@example.com",
            "subject": "Where is my refund?",
            "message": "I returned my order two weeks ago and still have no refund.",
        },
    )
    assert second.status_code == 201
    assert second.json()["prior_tickets"] == 1
    assert "written to us 1 time(s) before" in second.json()["reply"]


def test_post_with_no_identity_is_anonymous_and_never_a_repeat():
    """A ticket with no name or email is anonymous: no history is
    recorded, and a second stranger is never called a repeat
    customer (the old bug — every form ticket shared one default id)."""
    first = client.post(
        "/tickets",
        json={
            "subject": "Do you accept PayPal?",
            "message": "Hi, do you accept PayPal as a payment method?",
        },
    )
    assert first.status_code == 201
    assert first.json()["ticket"]["customer_id"] is None
    assert first.json()["prior_tickets"] == 0

    second = client.post(
        "/tickets",
        json={
            "subject": "Do you accept PayPal?",
            "message": "Hi, do you accept PayPal as a payment method?",
        },
    )
    assert second.status_code == 201
    assert second.json()["prior_tickets"] == 0
    assert "written to us" not in second.json()["reply"]


def test_post_with_name_only_uses_slug_customer_id():
    """A name without an email becomes a 'C-first-last' customer id."""
    post = client.post(
        "/tickets",
        json={
            "customer_name": "Jordan Lee",
            "subject": "Where is my refund?",
            "message": "I returned my order two weeks ago and still have no refund.",
        },
    )
    assert post.status_code == 201
    assert post.json()["ticket"]["customer_id"] == "C-jordan-lee"


def test_unknown_ticket_ids_return_404():
    assert client.post("/tickets", json={"ticket_id": "T-9999"}).status_code == 404
    assert client.get("/tickets/T-9999").status_code == 404


def test_post_with_no_ticket_and_no_text_returns_422():
    assert client.post("/tickets", json={}).status_code == 422
