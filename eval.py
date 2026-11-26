"""
eval.py — measure the desk on all 8 sample tickets (mock mode).

Classification accuracy: predicted category vs the "expected" label
in data/tickets.json.
Retrieval top-1 hit rate: is the top-ranked article one of the ticket's
expected articles?

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
          f"{'Top-1 article':<28} {'Hit':<4}")
    print("-" * 70)

    correct_cls = 0
    correct_ret = 0
    for ticket in tickets:
        result = desk.run(ticket, verbose=False, quiet=True)
        expected = ticket["expected"]
        predicted = result["classification"]["category"]
        cls_ok = predicted == expected["category"]
        top_title = (
            result["retrieved"][0]["title"] if result["retrieved"] else "none"
        )
        ret_ok = top_title in expected["articles"]
        correct_cls += cls_ok
        correct_ret += ret_ok
        print(f"{ticket['id']:<8} {expected['category']:<10} {predicted:<10} "
              f"{'OK' if cls_ok else 'MISS':<4} {top_title:<28} "
              f"{'OK' if ret_ok else 'MISS':<4}")

    print("-" * 70)
    total = len(tickets)
    print(f"Classification accuracy : {correct_cls}/{total} "
          f"({100 * correct_cls / total:.0f}%)")
    print(f"Retrieval top-1 hit rate: {correct_ret}/{total} "
          f"({100 * correct_ret / total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
