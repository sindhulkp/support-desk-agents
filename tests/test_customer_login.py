"""
tests/test_customer_login.py — demo customer sign-in and the
orders + refunds + past-messages view.

One sign-in shows a customer their orders (with refund status) and
their past messages with us, all in one place — that is the point:
most shops split those across three different pages. The accounts
are public demo accounts (password "demo1234", printed on the
homepage), not real security.
"""

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

DEMO_ACCOUNTS = [
    ("priya.sharma@example.com", "C-101", "Priya Sharma"),
    ("sofia.khan@example.com", "C-103", "Sofia Khan"),
    ("ava.brown@example.com", "C-107", "Ava Brown"),
]


@pytest.mark.parametrize("email,customer_id,name", DEMO_ACCOUNTS)
def test_login_valid_for_each_demo_account(email, customer_id, name):
    response = client.post(
        "/login", json={"email": email, "password": "demo1234"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == customer_id
    assert body["name"] == name
    assert body["email"] == email


def test_login_email_is_case_insensitive():
    response = client.post(
        "/login",
        json={"email": "Sofia.Khan@Example.com", "password": "demo1234"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "C-103"


@pytest.mark.parametrize(
    "email,password",
    [
        ("sofia.khan@example.com", "wrong-password"),  # right email, wrong password
        ("sofia.khan@example.com", ""),  # empty password
        ("nobody@example.com", "demo1234"),  # unknown email
        ("", "demo1234"),  # empty email
    ],
)
def test_login_wrong_credentials_return_401(email, password):
    response = client.post("/login", json={"email": email, "password": password})
    assert response.status_code == 401


def test_customer_orders_returns_only_that_customers_orders():
    """Priya has two orders (ORD-1001 Shipped, ORD-1005 Processing)
    and nothing else — no other customer's order ever leaks in."""
    response = client.get("/customers/C-101/orders")
    assert response.status_code == 200
    orders = response.json()
    assert [o["order_id"] for o in orders] == ["ORD-1001", "ORD-1005"]
    for order in orders:
        assert order["customer_name"] == "Priya Sharma"
        assert order["items"]  # items are listed
        assert order["total"] > 0
        assert order["status"]
        assert order["estimated_delivery"]
        assert order["carrier"] and order["tracking_number"]
        assert "refund" in order  # every order carries a refund field


def test_customer_orders_include_refund_status():
    """Sofia's delayed order has a refund requested; Ava's delivered
    order has a refund approved; Priya has no refund anywhere."""
    sofia = client.get("/customers/C-103/orders").json()
    assert len(sofia) == 1
    assert sofia[0]["order_id"] == "ORD-1003"
    assert sofia[0]["status"] == "Delayed"
    assert sofia[0]["refund"] == {
        "status": "requested",
        "amount": 79.99,
        "requested_date": "2026-11-26",
    }

    ava = client.get("/customers/C-107/orders").json()
    assert len(ava) == 1
    assert ava[0]["order_id"] == "ORD-1008"
    assert ava[0]["refund"]["status"] == "approved"
    assert ava[0]["refund"]["amount"] == 59.99
    assert "3–5 business days" in ava[0]["refund"]["note"]

    priya = client.get("/customers/C-101/orders").json()
    assert all(order["refund"] is None for order in priya)


def test_customer_orders_unknown_customer_is_empty():
    assert client.get("/customers/C-999/orders").json() == []


def test_customer_tickets_empty_safe_before_any_ticket():
    """A customer who has never written in gets an empty list —
    not an error, not someone else's tickets."""
    assert client.get("/customers/C-101/tickets").json() == []


def test_customer_tickets_show_processed_tickets():
    """After Sofia's sample ticket T-1003 is processed, her
    'My messages with us' list shows it — answered (the order
    lookup answered it), not passed to a human."""
    client.post("/tickets", json={"ticket_id": "T-1003"})
    response = client.get("/customers/C-103/tickets")
    assert response.status_code == 200
    tickets = response.json()
    assert len(tickets) == 1
    assert tickets[0]["id"] == "T-1003"
    assert tickets[0]["subject"] == "Order tracking stuck for 5 days"
    assert tickets[0]["status"] == "answered"
    assert tickets[0]["processed_at"]


def test_customer_tickets_match_tickets_written_in_by_email():
    """A ticket written in through the form is stored under the
    lowercased email, not the C- number — the customer's list must
    still find it, and another customer must not see it."""
    client.post(
        "/tickets",
        json={
            "customer_name": "Sofia Khan",
            "customer_email": "sofia.khan@example.com",
            "subject": "Do you accept PayPal?",
            "message": "Hi, do you accept PayPal as a payment method?",
        },
    )
    sofia = client.get("/customers/C-103/tickets").json()
    assert len(sofia) == 1
    assert sofia[0]["status"] == "passed to a human"  # no article answers it

    assert client.get("/customers/C-101/tickets").json() == []


def test_homepage_has_login_card_and_demo_credentials():
    """GET / shows the sign-in card with the demo accounts and the
    password printed on it, so anyone can try instantly."""
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "Sign in to track your orders" in text
    for email, _customer_id, name in DEMO_ACCOUNTS:
        assert email in text
        assert name in text
    assert "demo1234" in text
    assert "My orders" in text
    assert "My messages with us" in text


def test_homepage_has_no_project_explanation():
    """The page reads like a shop's support desk: none of the
    project-explanation wording may appear anywhere in it
    (checked case-insensitively — 'eval' also catches words
    like 'retrieval')."""
    text = client.get("/").text.lower()
    for banned in (
        "multi-agent",
        "classifier",
        "retriever",
        "responder",
        "relevance threshold",
        "33 tests",
        "built with python",
        "eval",
    ):
        assert banned not in text, f"homepage still says {banned!r}"
