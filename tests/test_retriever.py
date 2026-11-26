"""
tests/test_retriever.py — does BM25 find the right help article?

The retriever is the RAG half of the project: if it ranks the wrong
article first, the responder drafts a reply from the wrong facts.
These tests check the top-1 result against the expected labels in
data/tickets.json, plus the "no match" case (empty result, not a crash).
"""

from agents.retriever import BM25Retriever
from tools import get_ticket, knowledge_dir, load_tickets


def _retriever():
    return BM25Retriever(knowledge_dir())


def test_top1_matches_expected_article_for_three_tickets():
    """At least 3 tickets: the top-ranked article must be an expected one."""
    retriever = _retriever()
    tickets = load_tickets()
    checked = 0
    for ticket_id in ("T-1002", "T-1003", "T-1004"):
        ticket = get_ticket(tickets, ticket_id)
        results = retriever.search(f"{ticket['subject']} {ticket['message']}", top_k=3)
        assert results, f"no articles found for {ticket_id}"
        top_title = results[0][0]["title"]
        assert top_title in ticket["expected"]["articles"], (
            f"{ticket_id}: top-1 was {top_title!r}, "
            f"expected one of {ticket['expected']['articles']}"
        )
        checked += 1
    assert checked == 3


def test_all_tickets_retrieve_an_expected_article():
    """Stronger check: every sample ticket's top-1 is an expected article.

    Tickets with no expected articles (T-1009: nothing in the knowledge
    base answers it) are skipped here — for them, retrieving nothing is
    the correct behaviour, and test_uniqueness.py checks that.
    """
    retriever = _retriever()
    for ticket in load_tickets():
        if not ticket["expected"]["articles"]:
            continue
        results = retriever.search(f"{ticket['subject']} {ticket['message']}", top_k=3)
        assert results, f"no articles found for {ticket['id']}"
        assert results[0][0]["title"] in ticket["expected"]["articles"]


def test_no_match_returns_empty_list():
    """A query with no overlapping words returns [] — the responder then
    says 'I don't know' instead of guessing."""
    retriever = _retriever()
    results = retriever.search("zebra quantum trombone blorp", top_k=3)
    assert results == []


def test_weak_match_below_threshold_returns_empty_list():
    """A real question the knowledge base does not cover must also
    return [] — even though a word or two happens to match ("payment"
    appears in Payment Failed), the best score is below the
    relevance threshold, so the desk admits it has no answer instead
    of replying from a barely-related article."""
    retriever = _retriever()
    results = retriever.search("Do you accept Bitcoin as payment?", top_k=3)
    assert results == []
