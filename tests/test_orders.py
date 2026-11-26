"""
tests/test_orders.py — the real order lookup.

This is what makes the desk answer "where is MY order?" about an
actual order, instead of quoting a generic tracking article:
  1. The order number is extracted from the ticket text ("ORD-1001"
     or "#1001" — never a bare number like "2 orders").
  2. The order is looked up in data/orders.json.
  3. The reply states the order's REAL status, items, and delivery
     date — or, when the number does not exist, honestly says the
     order could not be found, with no invented status.
"""

from agents.orchestrator import SupportDesk
from agents.order_lookup import extract_order_id, find_order
from llm import MockLLM
from tools import get_ticket, knowledge_dir, load_tickets


def _desk():
    return SupportDesk(MockLLM(), knowledge_dir())


def _inline_ticket(ticket_id, subject, message, customer_id="C-999",
                   customer_name="Test Customer"):
    return {
        "id": ticket_id,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "subject": subject,
        "message": message,
    }


def test_extract_order_id_matches_both_forms():
    assert extract_order_id("Where is my order #1001?") == "ORD-1001"
    assert extract_order_id("My order ORD-1003 is late") == "ORD-1003"
    assert extract_order_id("ord-1002 please") == "ORD-1002"
    assert extract_order_id("Where is my order #9999?") == "ORD-9999"
    # A bare number is NOT an order id: "I have 2 orders" must never
    # trigger a lookup of some order 2.
    assert extract_order_id("I have 2 orders to check") is None
    assert extract_order_id("Please help with my delivery") is None


def test_find_order_returns_order_or_none():
    order = find_order("ORD-1001")
    assert order is not None
    assert order["order_id"] == "ORD-1001"
    assert order["status"] == "Shipped"
    assert order["customer_name"] == "Priya Sharma"
    assert find_order("ORD-9999") is None
    assert find_order(None) is None


def test_pipeline_with_real_order_id_states_real_status():
    """A ticket asking about a real order gets that order's real facts
    in the reply — status, items, delivery date — not a generic article."""
    ticket = _inline_ticket(
        "T-TEST-1", "Where is my order #1001?",
        "Hi, I ordered last week and I want to know where my order #1001 is.",
        customer_id="C-101", customer_name="Priya Sharma",
    )
    result = _desk().run(ticket, verbose=False, quiet=True)

    assert result["order_id_mentioned"] == "ORD-1001"
    assert result["order"]["order_id"] == "ORD-1001"
    assert result["order"]["status"] == "Shipped"
    assert "shipped" in result["reply"].lower()
    assert "Wireless Keyboard" in result["reply"]
    assert "2026-11-27" in result["reply"]  # the real estimated delivery


def test_pipeline_with_fake_order_id_says_not_found():
    """An order number that does not exist gets an honest not-found
    reply — and no invented status, tracking number, or date."""
    ticket = _inline_ticket(
        "T-TEST-2", "Where is my order #9999?",
        "Hi, where is my order #9999? I placed it last week.",
    )
    result = _desk().run(ticket, verbose=False, quiet=True)

    assert result["order_id_mentioned"] == "ORD-9999"
    assert result["order"] is None
    assert "couldn't find" in result["reply"].lower()
    reply = result["reply"].lower()
    for invented in ("shipped", "delivered", "out for delivery",
                     "processing", "delayed"):
        assert invented not in reply, f"reply invents status {invented!r}"


def test_sample_ticket_t1003_reports_its_real_delayed_order():
    """T-1003 names its order number (ORD-1003), and that order really
    is Delayed in the orders file — so the reply states the delay
    from the order itself, not from a generic article."""
    ticket = get_ticket(load_tickets(), "T-1003")
    result = _desk().run(ticket, verbose=False, quiet=True)

    assert result["order_id_mentioned"] == "ORD-1003"
    assert result["order"]["status"] == "Delayed"
    assert "delayed" in result["reply"].lower()


def test_answered_order_ticket_is_not_escalated():
    """The order lookup answered the ticket, so the empty-retrieval
    escalation must not fire — the reply stated the order's real
    status, and flagging it as 'no article answers this' would be a
    reason that is not true."""
    ticket = _inline_ticket(
        "T-TEST-3", "Where is my order #1001?",
        "Hi, where is my order #1001? I placed it last week.",
        customer_id="C-101", customer_name="Priya Sharma",
    )
    result = _desk().run(ticket, verbose=False, quiet=True)
    assert result["order"] is not None
    assert result["needs_human_review"] is False
    assert result["escalation_reason"] is None


def test_unanswered_order_ticket_is_escalated():
    """The not-found order ticket has no article and no order, so it
    goes to a human — the honest failure, not a guessed answer."""
    ticket = _inline_ticket(
        "T-TEST-4", "Where is my order #9999?",
        "Hi, where is my order #9999? I placed it last week.",
    )
    result = _desk().run(ticket, verbose=False, quiet=True)
    assert result["order"] is None
    assert result["needs_human_review"] is True
    assert result["escalation_reason"] is not None
