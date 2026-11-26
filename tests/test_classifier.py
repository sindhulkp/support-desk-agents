"""
tests/test_classifier.py — does the (mock) classifier return valid,
well-shaped results?

The classifier's output is data the code depends on, so the test checks
the SHAPE (exactly three fields, each from the allowed set) and the
CONTENT (a double-charge ticket is billing, a crash ticket is technical).
"""

import json

from agents.classifier import classify
from llm import MockLLM

CATEGORIES = {"billing", "technical", "shipping", "returns", "general"}
PRIORITIES = {"high", "normal", "low"}
SENTIMENTS = {"positive", "neutral", "negative"}


def test_mock_classifier_returns_valid_fields():
    llm = MockLLM()
    ticket = {
        "subject": "Charged twice",
        "message": "I was charged twice for my subscription, please refund me, this is urgent.",
    }
    result = classify(llm, ticket)
    assert set(result) == {"category", "priority", "sentiment"}
    assert result["category"] in CATEGORIES
    assert result["priority"] in PRIORITIES
    assert result["sentiment"] in SENTIMENTS


def test_mock_classifier_raw_output_is_valid_json():
    """The mock must speak the same JSON the real LLM is asked for."""
    llm = MockLLM()
    raw = llm.chat(
        [
            {"role": "system", "content": "You are a classifier. ROLE: classifier"},
            {"role": "user", "content": "TICKET:\nApp crashes\nThe app crashes on checkout."},
        ],
        temperature=0.0,
        json_mode=True,
    )
    parsed = json.loads(raw)
    assert set(parsed) == {"category", "priority", "sentiment"}


def test_classifier_labels_sample_tickets_correctly():
    llm = MockLLM()
    cases = {
        "T-1001": "billing",
        "T-1002": "technical",
        "T-1003": "shipping",
        "T-1004": "returns",
    }
    from tools import get_ticket, load_tickets

    tickets = load_tickets()
    for ticket_id, expected in cases.items():
        ticket = get_ticket(tickets, ticket_id)
        assert classify(llm, ticket)["category"] == expected
