"""
eval.py — measure the desk on all 9 sample tickets (mock mode).

Classification accuracy: predicted category vs the "expected" label
in data/tickets.json.
Retrieval top-1 hit rate: is the top-ranked article one of the ticket's
expected articles? For T-1009 (no article covers the question) the
correct result is NO articles retrieved — a hit means the retriever
stayed silent instead of forcing a weak match.
Honesty: for tickets with no expected articles, the reply must say it
does not know AND the ticket must be flagged for a human. That is the
behaviour that makes this desk different from a bot that answers
everything: it is measured, not just claimed.

This is a REPORT, not a gate: it prints the numbers honestly and always
exits 0. If a number is low, that is a finding to fix, not a failure to
hide. Run it after changing the classifier, the retriever, or the data.

Usage:
  python eval.py

Memory: the eval writes its ticket history to a throwaway temp file
instead of the real memory.json, so evaluating never pollutes your data.
"""

import os
import tempfile

import agents.orchestrator as orchestrator
from llm import MockLLM
from tools import knowledge_dir, load_tickets


def main():
    # Keep the eval's memory out of the real memory.json.
    orchestrator.MEMORY_FILE = os.path.join(
        tempfile.gettempdir(), "support-desk-eval-memory.json"
    )
    if os.path.exists(orchestrator.MEMORY_FILE):
        os.remove(orchestrator.MEMORY_FILE)

    desk = orchestrator.SupportDesk(MockLLM(), knowledge_dir())
    tickets = load_tickets()

    print(f"{'Ticket':<8} {'Expected':<10} {'Predicted':<10} {'Cls':<4} "
          f"{'Top-1 article':<28} {'Hit':<4} {'Esc':<4}")
    print("-" * 74)

    correct_cls = 0
    correct_ret = 0
    honest_ok = 0
    honest_total = 0
    for ticket in tickets:
        result = desk.run(ticket, verbose=False, quiet=True)
        expected = ticket["expected"]
        predicted = result["classification"]["category"]
        cls_ok = predicted == expected["category"]
        top_title = (
            result["retrieved"][0]["title"] if result["retrieved"] else "none"
        )
        if expected["articles"]:
            ret_ok = top_title in expected["articles"]
        else:
            # No article covers this ticket: retrieving nothing IS correct.
            ret_ok = len(result["retrieved"]) == 0
        correct_cls += cls_ok
        correct_ret += ret_ok
        esc = "Y" if result["needs_human_review"] else "-"
        if not expected["articles"]:
            honest_total += 1
            says_dont_know = (
                "don't have information" in result["reply"].lower()
                or "don't know" in result["reply"].lower()
            )
            if says_dont_know and result["needs_human_review"]:
                honest_ok += 1
        print(f"{ticket['id']:<8} {expected['category']:<10} {predicted:<10} "
              f"{'OK' if cls_ok else 'MISS':<4} {top_title:<28} "
              f"{'OK' if ret_ok else 'MISS':<4} {esc:<4}")

    print("-" * 74)
    total = len(tickets)
    print(f"Classification accuracy : {correct_cls}/{total} "
          f"({100 * correct_cls / total:.0f}%)")
    print(f"Retrieval top-1 hit rate: {correct_ret}/{total} "
          f"({100 * correct_ret / total:.0f}%)")
    if honest_total:
        print(f"Honesty (says I don't know + escalates): {honest_ok}/{honest_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
