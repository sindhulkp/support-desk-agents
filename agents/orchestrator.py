"""
agents/orchestrator.py — the boss agent that runs the whole pipeline.

Job: take one ticket and run the three agents in order:
  1. Classifier  — what kind of ticket is this?
  2. Retriever   — which help articles answer it?
  3. Responder   — draft a grounded reply.

This is the agent LOOP made visible: for every step we print
THINK (what the agent is deciding), ACT (what tool or call it makes)
and OBSERVE (what came back). In a bigger system the orchestrator could
also loop, retry, or branch — here it runs the three steps in sequence,
which is the simplest multi-agent pattern (a "pipeline" / "chain").

It also keeps per-customer memory: after each ticket it records the
ticket in memory.json, so the next time the same customer writes in,
the desk can see their history. Memory is what turns a one-off chatbot
into an agent that remembers.

Logging: every pipeline step also writes exactly one structured log line
(key=value pairs), so the run can be followed in a log file as well as on
screen. Set LOG_LEVEL=DEBUG for more detail.
"""

import json
import logging
import os

from .classifier import classify
from .responder import draft_reply
from .retriever import BM25Retriever

logger = logging.getLogger("support_desk")

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory.json")


def load_memory():
    """Read per-customer history from memory.json (empty dict if none yet)."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_memory(memory):
    """Write per-customer history back to memory.json."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def step(number, agent_name, think, act, observe):
    """Print one agent-loop step in a readable THINK / ACT / OBSERVE format."""
    print(f"\n--- Step {number}: {agent_name} ---")
    print(f"  THINK:   {think}")
    print(f"  ACT:     {act}")
    print(f"  OBSERVE: {observe}")


class SupportDesk:
    """Runs the 3-agent pipeline for one ticket at a time."""

    def __init__(self, llm, knowledge_dir):
        self.llm = llm
        self.retriever = BM25Retriever(knowledge_dir)

    def run(self, ticket, verbose=True, quiet=False):
        """Process one ticket end to end. Returns a result dict.

        verbose: print the THINK / ACT / OBSERVE steps (CLI demo).
        quiet:   print nothing at all (API and eval use this — the
                 structured log lines still record every step).
        """
        if not quiet:
            print(f"\n{'=' * 60}")
            print(f"Ticket {ticket['id']} — {ticket['customer_name']} "
                  f"({ticket['customer_id']})")
            print(f"Subject: {ticket['subject']}")
            print(f"Message: {ticket['message']}")
            print(f"{'=' * 60}")

        # -- Step 1: classify ------------------------------------------------
        classification = classify(self.llm, ticket)
        logger.info(
            "step=classify ticket_id=%s category=%s priority=%s sentiment=%s",
            ticket["id"], classification["category"],
            classification["priority"], classification["sentiment"],
        )
        if verbose and not quiet:
            step(
                1, "Classifier agent",
                "What kind of ticket is this, and how urgent?",
                "LLM call with JSON mode",
                f"category={classification['category']}, "
                f"priority={classification['priority']}, "
                f"sentiment={classification['sentiment']}",
            )

        # -- Step 2: retrieve ------------------------------------------------
        query = f"{ticket['subject']} {ticket['message']}"
        retrieved = self.retriever.search(query, top_k=3)
        top_title = retrieved[0][0]["title"] if retrieved else "none"
        top_score = retrieved[0][1] if retrieved else 0.0
        logger.info(
            "step=retrieve ticket_id=%s top_article=%r top_score=%.2f num_results=%d",
            ticket["id"], top_title, top_score, len(retrieved),
        )
        if verbose and not quiet:
            found = ", ".join(
                f"'{doc['title']}' (score {score:.2f})" for doc, score in retrieved
            ) or "nothing relevant found"
            step(
                2, "Retriever agent (BM25)",
                "Which help articles answer this ticket?",
                "BM25 keyword search over the knowledge base",
                f"top 3 articles: {found}",
            )

        # -- Step 3: respond -------------------------------------------------
        reply = draft_reply(self.llm, ticket, retrieved)
        logger.info(
            "step=respond ticket_id=%s reply_chars=%d",
            ticket["id"], len(reply),
        )
        if verbose and not quiet:
            step(
                3, "Responder agent",
                "Draft a reply using only the retrieved articles.",
                "LLM call, grounded in the articles with citations",
                "draft reply written (see below)",
            )

        if not quiet:
            print(f"\n>>> DRAFT REPLY (to {ticket['customer_name']}):\n{reply}\n")

        # -- Memory: remember this ticket for this customer ------------------
        memory = load_memory()
        history = memory.setdefault(ticket["customer_id"], [])
        history.append(
            {
                "ticket_id": ticket["id"],
                "category": classification["category"],
                "priority": classification["priority"],
            }
        )
        save_memory(memory)
        logger.info(
            "step=memory ticket_id=%s customer_id=%s total_tickets=%d",
            ticket["id"], ticket["customer_id"], len(history),
        )
        if verbose and not quiet:
            print(f"[memory] Saved to memory.json — customer "
                  f"{ticket['customer_id']} now has {len(history)} ticket(s) on file.")

        return {
            "ticket": ticket,
            "classification": classification,
            "retrieved": [
                {"title": doc["title"], "score": round(score, 2)}
                for doc, score in retrieved
            ],
            "reply": reply,
        }
