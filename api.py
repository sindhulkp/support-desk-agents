"""
api.py — the support desk as a web service (FastAPI).

The same 3-agent pipeline as main.py, but behind HTTP endpoints so any
other program can use it:

  GET  /health               — is the service up? Which LLM mode is it in?
  POST /tickets              — run the full pipeline on a ticket
  GET  /tickets/{ticket_id}  — fetch the stored result of a processed ticket

Run it:
  uvicorn api:app --reload
  curl http://127.0.0.1:8000/health

POST body — either run a sample ticket by id:
  {"ticket_id": "T-1001"}
or send a ticket inline (subject and message are required):
  {"id": "WEB-1", "customer_name": "Sam", "customer_email": "sam@example.com",
   "subject": "Where is my refund?", "message": "I returned my order..."}

How a customer is identified: an email (lowercased) becomes the
customer id, so the same person writing twice is recognised as a
repeat customer. A name alone becomes a "C-first-last" id. With
neither, the ticket is anonymous — it is processed, but no history
is recorded and the reply never claims a repeat visit that did not
happen. Sample tickets (ticket_id) keep their own customer ids.

How the LLM is chosen: if LLM_API_KEY is set in the environment (or a
.env file), the real LLM is used; otherwise the built-in mock LLM runs
the pipeline. Set MOCK_LLM=1 to force mock mode even with a key.
This is a learning project: there is no auth — do not expose it to the
internet as it is.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.orchestrator import SupportDesk
from config import settings, setup_logging
from llm import MockLLM, get_llm
from tools import get_ticket, knowledge_dir, load_env_file, load_tickets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "results.json")

setup_logging()
logger = logging.getLogger("support_desk")

app = FastAPI(
    title="Support Desk Agents",
    description="Multi-agent customer support desk with RAG.",
    version="0.2.0",
)

# -- in-memory store of processed results -----------------------------------
# Persisted to results.json so results survive a restart. This is a
# learning project: a real service would use a database.
_RESULTS = {}


def _load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            _RESULTS.update(json.load(f))


def _save_results():
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(_RESULTS, f, indent=2)


# -- the pipeline, built once and reused -------------------------------------
_DESK = None


def get_desk():
    """Build the SupportDesk on first use (lazy, so importing this module
    never crashes when no API key is set)."""
    global _DESK
    if _DESK is None:
        load_env_file()
        if settings.mock_llm or not settings.llm_api_key:
            logger.info("llm_mode=mock reason=no API key set (or MOCK_LLM=1)")
            llm = MockLLM()
            mode = "mock"
        else:
            logger.info("llm_mode=real model=%s", settings.llm_model)
            llm = get_llm(mock=False)
            mode = "real"
        _DESK = (SupportDesk(llm, knowledge_dir()), mode)
    return _DESK


def llm_mode():
    return get_desk()[1]


# -- request / response shapes ------------------------------------------------
class TicketRequest(BaseModel):
    """POST /tickets accepts either a sample ticket id or an inline ticket."""
    ticket_id: str | None = None       # e.g. "T-1001" — run a sample ticket
    id: str | None = None              # inline ticket id (default: auto)
    customer_id: str = "C-API"
    customer_name: str = "API Customer"
    customer_email: str | None = None  # when given, this identifies the customer
    subject: str = ""
    message: str = ""


def _customer_identity(body: TicketRequest):
    """Work out who an inline ticket is from.

    Priority: email (lowercased) > an explicitly sent customer_id >
    a slug of the name > nothing at all. "Nothing" is a real answer:
    an anonymous ticket gets customer_id None, so the pipeline records
    no history for it and can never mistake two strangers for one
    repeat customer.
    """
    email = (body.customer_email or "").strip().lower() or None
    name = (body.customer_name or "").strip() or None
    if email:
        return email, name or "Guest"
    if body.customer_id and body.customer_id != "C-API":
        return body.customer_id, name or "Guest"
    if name and name != "API Customer":
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return f"C-{slug}", name
    return None, "Guest"


# -- endpoints ----------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    """Serve the live demo page: pick a sample ticket (or write your own)
    and run it through the real pipeline in the browser. Same origin as
    the API, so the page needs no extra configuration or CORS."""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/health")
def health():
    """Liveness check + which LLM mode the service is running in."""
    _load_results()
    return {
        "status": "ok",
        "llm_mode": llm_mode(),
        "tickets_processed": len(_RESULTS),
    }


@app.post("/tickets", status_code=201)
def process_ticket(body: TicketRequest):
    """Run the full 3-agent pipeline on one ticket and store the result."""
    if body.ticket_id:
        ticket = get_ticket(load_tickets(), body.ticket_id)
        if ticket is None:
            raise HTTPException(
                status_code=404, detail=f"sample ticket {body.ticket_id} not found"
            )
    elif body.subject and body.message:
        customer_id, customer_name = _customer_identity(body)
        ticket = {
            "id": body.id or f"API-{len(_RESULTS) + 1}",
            "customer_id": customer_id,
            "customer_name": customer_name,
            "subject": body.subject,
            "message": body.message,
        }
    else:
        raise HTTPException(
            status_code=422,
            detail="send either ticket_id (e.g. 'T-1001') or subject + message",
        )

    desk, _mode = get_desk()
    # quiet=True: no console prints — the structured log lines record it.
    result = desk.run(ticket, verbose=False, quiet=True)
    result["processed_at"] = datetime.now(timezone.utc).isoformat()
    _load_results()
    _RESULTS[ticket["id"]] = result
    _save_results()
    logger.info("api=ticket_processed ticket_id=%s", ticket["id"])
    return result


@app.get("/tickets/{ticket_id}")
def get_result(ticket_id: str):
    """Fetch the stored pipeline result for a processed ticket."""
    _load_results()
    result = _RESULTS.get(ticket_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"no stored result for ticket {ticket_id}"
        )
    return result


@app.get("/stats")
def stats():
    """Live operations numbers for the dashboard, computed from the
    stored results: totals, breakdowns by category and priority, how
    many tickets were escalated, how many got an honest "I don't
    know", how many distinct customers have written in, and the most
    recent tickets. Empty when nothing has been processed yet — the
    dashboard fills up as tickets are run, it is never seeded with
    fake numbers."""
    _load_results()
    by_category: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    escalated = 0
    dont_know = 0
    customers = set()
    recent = []
    for ticket_id, result in _RESULTS.items():
        cat = result["classification"]["category"]
        by_category[cat] = by_category.get(cat, 0) + 1
        pri = result["classification"]["priority"]
        by_priority[pri] = by_priority.get(pri, 0) + 1
        if result.get("needs_human_review"):
            escalated += 1
        if not result.get("retrieved"):
            dont_know += 1
        # Anonymous tickets (customer_id None) are not a "customer seen".
        if result["ticket"].get("customer_id"):
            customers.add(result["ticket"]["customer_id"])
        recent.append({
            "id": ticket_id,
            "subject": result["ticket"]["subject"],
            "category": cat,
            "priority": pri,
            "escalated": bool(result.get("needs_human_review")),
        })
    return {
        "tickets_processed": len(_RESULTS),
        "by_category": by_category,
        "by_priority": by_priority,
        "escalated": escalated,
        "said_i_dont_know": dont_know,
        "customers_seen": len(customers),
        "recent": recent[-8:][::-1],
    }
