"""
tests/test_orchestrator.py — the whole pipeline end to end, in mock mode.

One ticket goes in; a draft reply comes out that cites a real article
from the knowledge base. That is the property that matters: the reply
is grounded in retrieved facts, not invented.
"""

from agents.orchestrator import SupportDesk
from llm import MockLLM
from tools import get_ticket, knowledge_dir, load_tickets


def test_end_to_end_mock_reply_cites_a_real_article():
    llm = MockLLM()
    desk = SupportDesk(llm, knowledge_dir())
    ticket = get_ticket(load_tickets(), "T-1004")  # damaged headphones

    result = desk.run(ticket, verbose=False, quiet=True)

    assert result["classification"]["category"] == "returns"
    assert result["retrieved"], "retriever found nothing"
    top_title = result["retrieved"][0]["title"]
    assert top_title == "Damaged Item"
    # The reply must quote the article it used (grounded generation).
    assert top_title in result["reply"]
    assert len(result["reply"]) > 50


def test_end_to_end_all_tickets_produce_replies():
    llm = MockLLM()
    desk = SupportDesk(llm, knowledge_dir())
    for ticket in load_tickets():
        result = desk.run(ticket, verbose=False, quiet=True)
        assert result["reply"], f"empty reply for {ticket['id']}"
        assert result["classification"]["category"] in {
            "billing", "technical", "shipping", "returns", "general",
        }
