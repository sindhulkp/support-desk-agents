"""
main.py — the command-line entry point for the support desk.

Usage:
  python main.py --demo                 Run 3 sample tickets with the
                                         built-in fake LLM (no API key).
  python main.py --ticket T-1001        Run one ticket with the real LLM
                                         (needs LLM_API_KEY in .env).
  python main.py --ticket T-1001 --mock Run one ticket with the fake LLM.

The output prints every agent step (THINK / ACT / OBSERVE) so you can
watch the multi-agent loop work, end to end.
"""

import argparse
import sys

from agents.orchestrator import SupportDesk
from config import setup_logging
from llm import get_llm
from tools import get_ticket, knowledge_dir, load_env_file, load_tickets

# Which tickets the demo runs (one per category, so you see variety).
DEMO_TICKETS = ["T-1001", "T-1003", "T-1004"]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent customer support desk with RAG."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true",
                       help="run 3 sample tickets with the mock LLM (no API key)")
    group.add_argument("--ticket", metavar="ID",
                       help="run one ticket by id, e.g. --ticket T-1001")
    parser.add_argument("--mock", action="store_true",
                        help="use the built-in fake LLM instead of a real API")
    args = parser.parse_args()

    # Load .env if present (sets LLM_API_KEY / LLM_BASE_URL / LLM_MODEL).
    load_env_file()

    # One structured log line per pipeline step (see config.py).
    setup_logging()

    tickets = load_tickets()

    if args.demo:
        print("DEMO MODE — using the built-in mock LLM (no API key needed).")
        llm = get_llm(mock=True)
        desk = SupportDesk(llm, knowledge_dir())
        for ticket_id in DEMO_TICKETS:
            ticket = get_ticket(tickets, ticket_id)
            desk.run(ticket)
        print("\nDemo finished. Try a real ticket with:")
        print("  python main.py --ticket T-1002 --mock")
        print("or with a real API key in .env:")
        print("  python main.py --ticket T-1002")
        return 0

    ticket = get_ticket(tickets, args.ticket)
    if ticket is None:
        print(f"Ticket {args.ticket} not found. Available ids:")
        print("  " + ", ".join(t["id"] for t in tickets))
        return 1

    try:
        llm = get_llm(mock=args.mock)
    except ValueError as e:
        # No API key and no --mock: explain instead of crashing.
        print(f"Error: {e}")
        print("Tip: run with --mock to use the fake LLM, or add a .env file.")
        return 1

    desk = SupportDesk(llm, knowledge_dir())
    desk.run(ticket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
