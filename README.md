# Support Desk Agents — a multi-agent customer support desk with RAG

A small, readable project that shows how **agentic AI** works in practice:
a support ticket comes in, and three specialised agents handle it in a
pipeline — one classifies it, one retrieves the relevant help articles,
and one drafts a grounded reply. The project also keeps per-customer
memory in a JSON file.

It is now a **real, runnable service**, not just a demo script: a FastAPI
API, 33 automated tests, an eval script that scores the pipeline on all
9 sample tickets, CI on every push, and a Dockerfile.

Built for a beginner learning agentic AI and prompt engineering. Every
file is commented, and `EXPLANATION.md` walks through the whole project
from scratch.

## What it does

```
Ticket arrives
    │
    ▼
┌──────────────┐   classify: category, priority, sentiment (LLM, JSON mode)
│ 1. Classifier│ ──▶ { category: "billing", priority: "high", sentiment: "negative" }
└──────┬───────┘
       ▼
┌──────────────┐   retrieve: BM25 keyword search over knowledge/*.md
│ 2. Retriever │ ──▶ top 3 articles: refund-policy, subscription-cancel, ...
└──────┬───────┘
       ▼
┌──────────────┐   respond: draft a reply using ONLY those articles,
│ 3. Responder │   with citations [1] [2] — or say "I don't know"
└──────┬───────┘
       ▼
  Draft reply + memory saved to memory.json
```

**Key ideas demonstrated:** multi-agent orchestration, the agent loop
(THINK → ACT → OBSERVE), RAG (retrieval-augmented generation), structured
JSON output, grounded generation with citations, per-customer memory,
structured logging, a tested HTTP API — and a **real order lookup**: when
a ticket mentions an order number (`ORD-1001` or `#1001`), the desk looks
it up in `data/orders.json` and answers from the order's real status,
items, and delivery date, or honestly says the order could not be found.

## What makes this different from other support-bot projects

Most demos answer every ticket. This one is built around the moments
that matter more — and each one is measured, not just claimed:

- **It says "I don't know" — and means it.** The retriever has a
  relevance threshold (measured: real tickets score 15.49–36.45,
  off-topic questions at most 9.18). Below it, no article is returned,
  the responder refuses to guess, and the ticket is flagged for a
  human. Ticket T-1009 ("Do you accept Bitcoin?") proves it.
- **It knows when to hand off to a human.** A ticket is escalated
  when no article answers it, or when an unhappy customer with an
  urgent problem has already written twice before. Everything else
  gets a draft reply; the hard cases get a person.
- **It remembers — and uses it.** Memory is not just stored. The
  reply acknowledges a returning customer, and the escalation rule
  reads their history. A repeat complaint is treated differently
  from a first contact.
- **It looks up real orders by number.** A ticket asking "Where is my
  order #1001?" gets that order's real status, items, and delivery
  date from `data/orders.json` — not a generic tracking article. A
  number that does not exist (`#9999`) gets an honest "I couldn't
  find that order", never an invented status. Ticket T-1003 names its
  real order, ORD-1003, which is Delayed — and the reply says so.
- **It is measured three ways.** 33 tests, an eval (9/9
  classification, 9/9 retrieval, honesty 1/1, order lookup 2/2 in
  mock mode), and CI on every push — so "it works" is a number, not a
  feeling.

## Setup

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. (Optional) set up a real API key — only needed for real-LLM mode
cp .env.example .env
# then edit .env and paste your key
```

For development (tests), also install the dev dependencies:

```bash
pip install -r dev-requirements.txt
```

> If `python` is not found on your machine, use `python3` instead in
> every command (macOS/Linux often only have `python3`).

## Run it (CLI)

```bash
# Demo mode — no API key needed, uses the built-in mock LLM
python main.py --demo

# One ticket with the mock LLM
python main.py --ticket T-1001 --mock

# One ticket with a real LLM (needs LLM_API_KEY in .env)
python main.py --ticket T-1001

# See all tickets
python -c "import json; [print(t['id'], '-', t['subject']) for t in json.load(open('data/tickets.json'))]"
```

Sample tickets: `T-1001` double charge, `T-1002` password reset,
`T-1003` stuck tracking (names its real order, ORD-1003, which is
Delayed), `T-1004` damaged headphones, `T-1005` refund
status, `T-1006` app crash, `T-1007` address change, `T-1008` late return,
`T-1009` Bitcoin question (no article covers it — the "I don't know" case).

Real orders live in `data/orders.json` (8 orders, ORD-1001 to ORD-1008).
Ask about one by number — `Where is my order #1001?` — and the desk
answers from that order's real status. `#9999` does not exist, and the
desk says so honestly instead of inventing a status.

## Run it (API)

The same pipeline as a web service:

```bash
uvicorn api:app --reload
```

```bash
# Health check — also tells you which LLM mode is running
curl http://127.0.0.1:8000/health
# {"status": "ok", "llm_mode": "mock", "tickets_processed": 0}

# Process a sample ticket (runs classifier -> retriever -> responder)
curl -X POST http://127.0.0.1:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "T-1001"}'

# Process a ticket sent inline
curl -X POST http://127.0.0.1:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"id": "WEB-1", "customer_name": "Sam", "customer_email": "sam@example.com",
       "subject": "Where is my refund?",
       "message": "I returned my order two weeks ago and still have no refund."}'

# Ask about a real order by number — the reply states its real status
curl -X POST http://127.0.0.1:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"subject": "Where is my order #1001?",
       "message": "Hi, where is my order #1001? I placed it last week."}'
# ... "order": {"order_id": "ORD-1001", "status": "Shipped", ...}

# An order number that does not exist is reported honestly
curl -X POST http://127.0.0.1:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"subject": "Where is my order #9999?",
       "message": "Hi, where is my order #9999?"}'
# ... "order": null, "order_id_mentioned": "ORD-9999", reply: "I couldn't find..."

# Fetch the stored result of a processed ticket
curl http://127.0.0.1:8000/tickets/T-1001

# Live operations numbers for the dashboard
curl http://127.0.0.1:8000/stats
```

Without an API key the API runs in mock mode (same fake LLM as
`--demo`). With `LLM_API_KEY` in `.env` it uses the real LLM.
Interactive docs are at `http://127.0.0.1:8000/docs`.
Open `http://127.0.0.1:8000/` in a browser for the live demo page and
operations dashboard — it talks to the same API, no extra setup.

## Tests and eval

```bash
# 33 tests: retriever ranking, classifier shape, end-to-end pipeline, API,
# order lookup, customer identity — plus the behaviours that make this
# desk different (honesty, memory, escalation)
python -m pytest -q

# Eval: run all 9 tickets in mock mode and score them, plus 2 order probes
python eval.py
# Classification accuracy : 9/9 (100%)
# Retrieval top-1 hit rate: 9/9 (100%)
# Honesty (says I don't know + escalates): 1/1
# Order lookup: 2/2
```

Each ticket in `data/tickets.json` carries an `"expected"` label
(category + the articles that should answer it), so the numbers are
measured, not eyeballed. `eval.py` writes its memory to a temp file,
so evaluating never touches the real `memory.json`.

## Docker

```bash
docker build -t support-desk .
docker run -p 8000:8000 support-desk
# With a real key:
docker run -p 8000:8000 -e LLM_API_KEY=your_key support-desk
```

## CI

`.github/workflows/ci.yml` runs on every push: install dependencies,
run pytest, run `python main.py --demo`. Everything it runs uses the
mock LLM, so CI needs no API key and no secrets.

## Using Google Gemini (free key) instead of OpenAI

Gemini has an OpenAI-compatible endpoint, so the same code works — just
point the base URL and model at Gemini in your `.env`:

```env
LLM_API_KEY=your-free-gemini-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.0-flash
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).
No code changes needed — `LLM_BASE_URL` and `LLM_MODEL` are the only
things that change.

## How each agent works

| Agent | File | Input | Output | Technique |
|---|---|---|---|---|
| Classifier | `agents/classifier.py` | ticket text | `{category, priority, sentiment}` | LLM + JSON mode |
| Retriever | `agents/retriever.py` | ticket text | top 3 help articles + scores | BM25 (pure Python) |
| Order lookup | `agents/order_lookup.py` | order id from the ticket | the real order, or `None` | regex + JSON file lookup |
| Responder | `agents/responder.py` | ticket + articles + order | draft reply with citations | LLM, grounded-only prompt |
| Orchestrator | `agents/orchestrator.py` | ticket | full result + saved memory | runs 1→2→3 in sequence |

## Project layout

```
support-desk-agents/
├── main.py                 # CLI entry point (--demo / --ticket)
├── api.py                  # FastAPI service: /health, POST /tickets, GET /tickets/{id}, GET /stats
├── eval.py                 # scores the pipeline on all 9 tickets + 2 order probes (mock mode)
├── config.py               # central settings from env vars + logging setup
├── llm.py                  # LLM wrapper + mock LLM (the only API code)
├── tools.py                # loads tickets, knowledge base, .env
├── agents/
│   ├── __init__.py
│   ├── classifier.py       # agent 1: category / priority / sentiment
│   ├── retriever.py        # agent 2: pure-Python BM25 search
│   ├── order_lookup.py     # tool: extract an order id, look it up in orders.json
│   ├── responder.py        # agent 3: grounded draft reply (articles + order)
│   └── orchestrator.py     # runs the 3 agents + order lookup + per-customer memory + logging
├── tests/                  # pytest: retriever, classifier, orchestrator, API, orders
│   ├── conftest.py         # temp memory/results files so tests stay clean
│   ├── test_retriever.py
│   ├── test_classifier.py
│   ├── test_orchestrator.py
│   ├── test_orders.py
│   └── test_api.py
├── .github/workflows/ci.yml  # CI: pytest + demo on every push
├── Dockerfile              # container for the API (python:3.12-slim)
├── data/
│   ├── tickets.json        # 9 sample tickets, each with an "expected" label
│   └── orders.json         # 8 real orders (ORD-1001–ORD-1008) with live statuses
├── knowledge/              # 7 help articles (the RAG knowledge base)
│   ├── refund-policy.md
│   ├── password-reset.md
│   ├── delivery-tracking.md
│   ├── damaged-item.md
│   ├── subscription-cancel.md
│   ├── payment-failed.md
│   └── app-crashes.md
├── memory.json             # created on first run: per-customer history
├── results.json            # created by the API: stored ticket results
├── requirements.txt        # openai, fastapi, uvicorn
├── dev-requirements.txt    # pytest, httpx (tests only)
├── .env.example            # copy to .env, add your key (never commit .env)
├── README.md               # this file
└── EXPLANATION.md          # from-scratch walkthrough + manager Q&A
```

## Adding your own data

- **New help article:** drop a `.md` file into `knowledge/` — the
  retriever picks it up automatically, no code changes.
- **New ticket:** add an entry to `data/tickets.json` with `id`,
  `customer_id`, `customer_name`, `subject`, `message`, and an
  `"expected"` label (`category` + `articles`) so the eval can score it.
- **New order:** add an entry to `data/orders.json` with `order_id`
  (form `ORD-1009`), `customer_id`, `customer_name`, `email`, `items`
  (name, qty, price), `total`, `status` (Processing, Shipped, Out for
  delivery, Delivered, or Delayed), `order_date`, `estimated_delivery`,
  `carrier`, `tracking_number`, and `city`. The lookup picks it up
  with no code changes — ask about it by number and the desk answers
  from its real status.

## Notes and limits

- The mock LLM in demo mode is rule-based (keyword matching). It shows
  the pipeline working, but its answers are simple — the real LLM gives
  much better replies.
- BM25 is keyword search, not semantic search: it matches words, not
  meanings. A ticket saying "money taken twice" won't match an article
  that only says "duplicate charge" unless the words overlap. That is a
  real limitation worth mentioning to your manager — and a natural next
  step is embedding-based (vector) retrieval. The relevance threshold
  (10.0) softens this: a weak keyword coincidence no longer produces
  an answer from the wrong article, it produces "I don't know".
  Short, heavily paraphrased questions can score below 10.0 even when
  answerable — the safe failure is "I don't know" + escalation, never
  a wrong answer. Note: a ticket about a real order is answered by the
  order lookup even when no article scores above the threshold.
- The order lookup matches an exact order number (`ORD-1001` or
  `#1001`). A bare number ("I have 2 orders") is deliberately not
  treated as an order id, and the orders "database" is a JSON file —
  fine for a demo; a production service would use a real database.
- The API has no auth and the "database" is a local JSON file
  (`results.json` / `memory.json`). Fine for a demo; a production
  service would add auth and a real database.
