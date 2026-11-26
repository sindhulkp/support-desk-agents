"""
agents/orchestrator.py — the boss agent that runs the whole pipeline.

Job: take one ticket and run the three agents in order:
  1. Classifier  — what kind of ticket is this?
  2. Retriever   — which help articles answer it?
  3. Responder   — draft a grounded reply.

Between steps 2 and 3 it also looks up any order number mentioned in
the ticket — a real tool call against data/orders.json — so the reply
can state the order's REAL status, items, and delivery date instead of
quoting a generic article. When the number does not exist, the reply
says the order could not be found; it never invents a status.

This is the agent LOOP made visible: for every step we print
THINK (what the agent is deciding), ACT (what tool or call it makes)
and OBSERVE (what came back). In a bigger system the orchestrator could
also loop, retry, or branch — here it runs the three steps in sequence,
which is the simplest multi-agent pattern (a "pipeline" / "chain").

It also keeps per-customer memory: after each ticket it records the
ticket in memory.json, so the next time the same customer writes in,
the desk can see their history — and USES it: the reply acknowledges
a repeat customer, and a repeat unhappy customer with an urgent
problem is escalated to a human instead of getting another auto-reply.
Memory is what turns a one-off chatbot into an agent that remembers.

Finally, the orchestrator decides when NOT to answer: if no help
article really answers the ticket (the retriever's relevance
threshold), the desk says "I don't know" and flags the ticket for a
human, rather than drafting a reply from the wrong article.

Logging: every pipeline step also writes exactly one structured log line
(key=value pairs), so the run can be followed in a log file as well as on
screen. Set LOG_LEVEL=DEBUG for more detail.
"""

import json
import logging
import os

from .classifier import classify
from .order_lookup import extract_order_id, find_order
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
            print(f"Ticket {ticket['id']} — {ticket.get('customer_name', 'Guest')} "
                  f"({ticket.get('customer_id') or 'anonymous'})")
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

        # -- Order lookup: a real tool call, not a guess ---------------------
        # If the ticket mentions an order number ("ORD-1001" or "#1001"),
        # look it up in the shop's orders file. The reply can then state
        # the order's REAL status, items, and delivery date. When the
        # number does not exist, order is None and the responder must
        # say so honestly instead of inventing a status.
        order_id_mentioned = extract_order_id(
            f"{ticket['subject']} {ticket['message']}"
        )
        order = (
            find_order(order_id_mentioned) if order_id_mentioned else None
        )
        logger.info(
            "step=order_lookup ticket_id=%s order_id=%s found=%s",
            ticket["id"], order_id_mentioned, order is not None,
        )
        if verbose and not quiet:
            if order_id_mentioned:
                found = (
                    f"found: {order['status']}, "
                    f"estimated delivery {order['estimated_delivery']}"
                    if order else "no such order in the system"
                )
                print(f"[order] Order {order_id_mentioned} mentioned — {found}.")
            else:
                print("[order] No order number mentioned — lookup skipped.")

        # -- Memory: read this customer's history BEFORE replying ---------
        # Most support bots save memory and never use it. Here the reply
        # is written knowing the customer's history: a repeat customer
        # gets acknowledged, and a repeat angry customer gets escalated
        # to a human instead of another auto-reply.
        # A ticket with no customer identity (customer_id None — a
        # stranger who left the name/email fields blank on the web
        # form) has no history and saves none: every such ticket is a
        # first contact, never a false "repeat customer".
        memory = load_memory()
        customer_id = ticket.get("customer_id")
        history = memory.get(customer_id, []) if customer_id else []
        prior_tickets = len(history)
        prior_categories = [h["category"] for h in history]
        logger.info(
            "step=memory ticket_id=%s customer_id=%s prior_tickets=%d",
            ticket["id"], customer_id, prior_tickets,
        )

        # -- Step 3: respond -------------------------------------------------
        reply = draft_reply(
            self.llm, ticket, retrieved,
            prior_tickets=prior_tickets,
            prior_categories=prior_categories,
            order=order,
            order_id_mentioned=order_id_mentioned,
        )
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
            print(f"\n>>> DRAFT REPLY (to {ticket.get('customer_name', 'Guest')}):\n{reply}\n")

        # -- Escalation: know when NOT to auto-reply -------------------------
        # A bot that answers everything is not helpful — it is risky.
        # This ticket goes to a human when:
        #   * no help article answers it AND the order lookup did not
        #     answer it either (the honest "I don't know" case), or
        #   * an unhappy customer with an urgent problem has already
        #     written twice before (a repeat complaint needs a person).
        # An order the lookup found counts as answered: the reply
        # stated the order's real status, so the empty-retrieval rule
        # must not fire — otherwise a correctly answered order ticket
        # would be flagged with a reason that is not true.
        needs_human_review = False
        escalation_reason = None
        if not retrieved and order is None:
            needs_human_review = True
            escalation_reason = (
                "No help article answers this ticket — the desk said "
                "'I don't know' instead of guessing."
            )
        elif (
            classification["priority"] == "high"
            and classification["sentiment"] == "negative"
            and prior_tickets >= 2
        ):
            needs_human_review = True
            escalation_reason = (
                f"Repeat customer ({prior_tickets} prior tickets) with an "
                "urgent, unhappy ticket — needs a human, not another auto-reply."
            )
        logger.info(
            "step=escalate ticket_id=%s needs_human_review=%s",
            ticket["id"], needs_human_review,
        )
        if verbose and not quiet:
            if prior_tickets > 0:
                print(f"[memory] Customer {customer_id} has "
                      f"{prior_tickets} prior ticket(s) on file "
                      f"({', '.join(prior_categories)}).")
            if needs_human_review:
                print(f"[escalation] HUMAN REVIEW NEEDED — {escalation_reason}")
            else:
                print("[escalation] No escalation — safe to send the draft reply.")

        # -- Memory: record this ticket for next time ------------------------
        # Only when the ticket carries a customer identity. An anonymous
        # web-form ticket (customer_id None) is never recorded, so two
        # strangers can never be mistaken for one repeat customer.
        if customer_id:
            history.append(
                {
                    "ticket_id": ticket["id"],
                    "category": classification["category"],
                    "priority": classification["priority"],
                }
            )
            memory[customer_id] = history
            save_memory(memory)
            logger.info(
                "step=memory ticket_id=%s customer_id=%s total_tickets=%d",
                ticket["id"], customer_id, len(history),
            )
            if verbose and not quiet:
                print(f"[memory] Saved to memory.json — customer "
                      f"{customer_id} now has {len(history)} ticket(s) on file.")
        else:
            logger.info(
                "step=memory ticket_id=%s customer_id=None skipped (anonymous)",
                ticket["id"],
            )

        return {
            "ticket": ticket,
            "classification": classification,
            "retrieved": [
                {"title": doc["title"], "score": round(score, 2)}
                for doc, score in retrieved
            ],
            "reply": reply,
            "prior_tickets": prior_tickets,
            "order": order,
            "order_id_mentioned": order_id_mentioned,
            "needs_human_review": needs_human_review,
            "escalation_reason": escalation_reason,
        }
