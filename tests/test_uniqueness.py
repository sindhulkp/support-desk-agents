"""
tests/test_uniqueness.py — the behaviours that make this desk different
from a typical support chatbot demo.

Most demos answer every ticket. This one is measured on the moments
that matter more:
  1. It says "I don't know" (and escalates) when no help article
     answers the question, instead of replying from a weak match.
  2. It remembers the customer: a repeat customer is acknowledged,
     and a repeat unhappy customer with an urgent problem goes to a
     human instead of getting another auto-reply.
  3. A first-time customer with a normal problem is NOT escalated —
     the desk only interrupts a human when it should.
"""

from agents.orchestrator import SupportDesk
from agents.retriever import BM25Retriever
from llm import MockLLM
from tools import get_ticket, knowledge_dir, load_tickets


def _desk():
    return SupportDesk(MockLLM(), knowledge_dir())


def test_off_topic_ticket_says_i_dont_know_and_escalates():
    """T-1009 asks about Bitcoin — no help article covers it."""
    ticket = get_ticket(load_tickets(), "T-1009")
    result = _desk().run(ticket, verbose=False, quiet=True)

    assert result["retrieved"] == [], "a weak match should not be returned"
    assert "don't have information" in result["reply"].lower()
    assert result["needs_human_review"] is True
    assert result["escalation_reason"] is not None


def test_repeat_angry_customer_escalates_but_first_contact_does_not():
    """Priya (C-101) writes about a double charge, high priority and
    negative. The first time, the desk drafts a reply. By her third
    ticket about it, a human needs to step in."""
    desk = _desk()
    ticket = get_ticket(load_tickets(), "T-1001")

    first = desk.run(ticket, verbose=False, quiet=True)
    assert first["needs_human_review"] is False
    assert first["prior_tickets"] == 0

    desk.run(ticket, verbose=False, quiet=True)  # second contact
    third = desk.run(ticket, verbose=False, quiet=True)
    assert third["prior_tickets"] == 2
    assert third["needs_human_review"] is True
    assert "Repeat customer" in third["escalation_reason"]


def test_reply_acknowledges_a_repeat_customer():
    """The second time Priya writes, the reply opens by recognising
    her — memory is used, not just stored."""
    desk = _desk()
    ticket = get_ticket(load_tickets(), "T-1001")

    desk.run(ticket, verbose=False, quiet=True)
    second = desk.run(ticket, verbose=False, quiet=True)

    assert second["prior_tickets"] == 1
    assert "written to us 1 time(s) before" in second["reply"]


def test_first_time_customer_with_normal_problem_is_not_escalated():
    """Sofia's stuck delivery (T-1003) is a normal first contact:
    the desk answers it and does not bother a human."""
    result = _desk().run(
        get_ticket(load_tickets(), "T-1003"), verbose=False, quiet=True
    )
    assert result["needs_human_review"] is False
    assert result["escalation_reason"] is None
    assert result["prior_tickets"] == 0


def test_retriever_threshold_rejects_weak_matches():
    """Direct check on the search: the Bitcoin question's best score
    (2.43) is below the 10.0 relevance threshold."""
    retriever = BM25Retriever(knowledge_dir())
    assert retriever.search("Do you accept Bitcoin as payment?") == []
