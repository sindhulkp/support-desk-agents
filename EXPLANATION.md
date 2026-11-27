# EXPLANATION.md — the whole project, from scratch

Written for someone who has to explain this project to their manager.
Read it top to bottom once, then use the "2-minute demo" and the
"Manager questions" sections when you present it.

---

## 1. What is agentic AI?

**One sentence:** an AI agent is a program that takes a goal, decides
for itself what steps to take, uses tools to gather information or do
things, and produces a result — instead of just answering one question.

A normal chatbot is one question → one answer. An **agent** works in a
**loop**:

```
THINK → ACT → OBSERVE → (repeat) → final answer
```

- **THINK** — look at the situation and decide what to do next.
- **ACT** — do it: call a tool, search a database, ask the LLM something.
- **OBSERVE** — look at what came back, and decide the next step.

**Multi-agent** means several small agents, each good at one job, working
together instead of one giant agent doing everything. That is what this
project is.

## 2. What this project does (in plain English)

A customer writes a support ticket, for example: *"I was charged twice
for my subscription, please refund the extra charge."*

Instead of one prompt that tries to do everything, three agents handle
the ticket in a pipeline:

1. **Classifier** — reads the ticket and decides: this is a *billing*
   problem, *high* priority, and the customer sounds *negative*
   (angry). It answers in JSON so the code can use the result.
2. **Retriever** — searches the help articles (the knowledge base) for
   the ones that answer this ticket. It uses BM25, a keyword search
   algorithm written in pure Python. For the double-charge ticket it
   finds the refund policy and the subscription-cancellation article.
3. **Responder** — writes the reply to the customer, but it is only
   allowed to use facts from the articles the retriever found. It cites
   them as [1] and [2]. If no article answers the question, it says
   "I don't know" and hands the ticket to a human — it never invents a
   policy.

Between steps 2 and 3 there is one more tool call: the **order
lookup**. If the ticket mentions an order number ("ORD-1001" or
"#1001"), the desk looks it up in `data/orders.json` — a small
database of real orders with live statuses — and the reply states
that order's real status, items, and delivery date. If the number
does not exist, the reply honestly says the order could not be
found. That is what makes "Where is my order?" answerable about an
actual order, instead of quoting a generic tracking article.

Afterwards the system saves the ticket in `memory.json`, so the next
time the same customer writes in, the desk knows their history.

## 2A. Customer sign-in: orders, refunds, and past messages in one place

On the website, a customer can sign in and see everything about
their orders in one view — each order's status, items, total,
delivery date, carrier and tracking, its refund status, and below
that, their past messages with us and whether each one was answered
or passed to a human. Most shops split those three things across
three different pages; here one sign-in shows them together. A
delayed order says what happens next in plain language instead of
just showing the word "Delayed".

There is no sign-up: three demo accounts are printed on the sign-in
card, all with the password `demo1234`, and clicking one signs you
straight in:

- **priya.sharma@example.com** — Priya Sharma: a Shipped order and a Processing order, no refund.
- **sofia.khan@example.com** — Sofia Khan: a Delayed order (ORD-1003) with a refund requested.
- **ava.brown@example.com** — Ava Brown: a Delivered order (ORD-1008) with a refund approved.

The refund data lives on the order itself in `data/orders.json`
(each order has a `"refund"` field: `null` when there is no refund,
or the status, amount, and dates when there is). The page never
invents a refund where none exists — an order with no refund says
so plainly. The sign-in is demo-only: one shared public password,
no password storage, no real accounts. Production would need real
authentication; this is here so anyone can try the tracking view
instantly. The endpoints are `POST /login`,
`GET /customers/{id}/orders`, and `GET /customers/{id}/tickets` in
`api.py`, and the no-sign-in path still works exactly as before —
writing "Where is my order #1001?" in the message box still looks
the order up and answers with its real status.

## 3. The agent loop, exactly as it happens here

Run `python main.py --demo` and you will see each step printed as
THINK / ACT / OBSERVE. Here is what happens for ticket T-1001
("Charged twice for my subscription"):

```
Step 1: Classifier agent
  THINK:   What kind of ticket is this, and how urgent?
  ACT:     LLM call with JSON mode
  OBSERVE: category=billing, priority=high, sentiment=negative

Step 2: Retriever agent (BM25)
  THINK:   Which help articles answer this ticket?
  ACT:     BM25 keyword search over the knowledge base
  OBSERVE: top 3 articles: 'Cancelling Your Subscription' (score 20.27),
           'Refund Policy' (score 17.98) — only two score above the 10.0
           relevance threshold, so only two are returned

Step 3: Responder agent
  THINK:   Draft a reply using only the retrieved articles.
  ACT:     LLM call, grounded in the articles with citations
  OBSERVE: draft reply written

>>> DRAFT REPLY ...
[memory] Saved to memory.json — customer C-101 now has 1 ticket(s) on file.
```

Note the design: the classifier's output is never shown to the customer
— it is internal routing information. The retriever's output is never
shown either — it is evidence for the responder. Only the final reply
leaves the system. That separation is the core idea of a pipeline.

## 4. RAG — the most important concept here

**RAG = Retrieval-Augmented Generation.** Two steps:

1. **Retrieval** — before answering, fetch the relevant facts from your
   own documents (here: the 7 help articles).
2. **Generation** — give those facts to the LLM and let it write the
   answer using only them.

Why it matters: an LLM on its own will happily invent a refund policy
("refunds take 2 days!") that sounds confident and is wrong. RAG
grounds the answer in documents you control. In this project the
responder's system prompt says: *"Use ONLY the information in the help
articles... If the articles do not contain the answer, say clearly that
you do not have that information."* That one instruction is what stops
hallucinations.

## 5. BM25 — how the retriever works (no magic)

BM25 is a keyword ranking formula from the 1990s, and it is still the
baseline every search system is compared against. It scores each article
against the ticket's words:

- **Term frequency (TF):** a word appearing often in an article raises
  its score — but with diminishing returns (appearing 10 times is not
  10× better than appearing once).
- **Inverse document frequency (IDF):** rare words count more. The word
  "refund" appears in 2 of 7 articles, so it is informative; the word
  "the" appears in all 7, so it is worth almost nothing.
- **Length normalisation:** long articles don't win just for being long;
  scores are adjusted by article length relative to the average.

Formula (one line per idea):

```
IDF(word) = ln(1 + (N - docs_with_word + 0.5) / (docs_with_word + 0.5))
score(doc) = Σ over query words: IDF × (TF × 2.5) / (TF + 1.5 × (0.25 + 0.75 × len / avg_len))
```

The whole implementation is ~60 lines in `agents/retriever.py`, pure
Python, no libraries. That is deliberate: you can explain every line.

**Honest limitation:** BM25 matches *words*, not *meaning*. A ticket
saying "money taken twice" will not match an article that only says
"duplicate charge" — different words, zero overlap. The industry fix is
embedding-based (vector) search, which matches meaning. That is the
natural "next step" for this project (see §9).

## 6. File by file

| File | What it does | Key idea to remember |
|---|---|---|
| `main.py` | Command line: `--demo`, `--ticket T-1001`, `--mock` | The entry point; parses arguments and starts the desk |
| `api.py` | FastAPI service: `GET /health`, `POST /tickets`, `GET /tickets/{id}`, `GET /stats`, plus `POST /login` and `GET /customers/{id}/orders` and `/tickets` for the sign-in view | The same pipeline, behind HTTP, so other programs can use it |
| `eval.py` | Runs all 9 tickets in mock mode, prints accuracy + hit rate + honesty + order lookup | Measurement: proves the pipeline works instead of claiming it |
| `config.py` | Central settings from env vars + `setup_logging()` | One place for config; values read on access, never hardcoded |
| `llm.py` | `LLMClient` (real API) + `MockLLM` (fake, rule-based) | Only file that talks to an LLM; config comes from env vars only |
| `tools.py` | Loads `tickets.json`, finds the knowledge folder, loads `.env` | Small helpers; "tools" are how agents touch the world |
| `agents/classifier.py` | Sends ticket → LLM, asks for JSON | Structured output: JSON mode makes the answer machine-readable |
| `agents/retriever.py` | `BM25Retriever` class | Pure-Python BM25; index built once at startup |
| `agents/order_lookup.py` | Extracts an order id, looks it up in `orders.json` | A real tool: the desk answers from an actual order's status |
| `agents/responder.py` | Sends ticket + articles + order → LLM | Grounded generation: only the retrieved facts, with citations |
| `agents/orchestrator.py` | `SupportDesk.run()` — steps 1→2→3 + order lookup + memory + log lines | The pipeline; prints THINK/ACT/OBSERVE, logs one line per step |
| `tests/` | pytest: retriever, classifier, orchestrator, API, orders | Tests pin the behaviour, so changes can't silently break it |
| `.github/workflows/ci.yml` | CI: install, pytest, demo on every push | Automation: every push is checked, no key needed (mock mode) |
| `Dockerfile` | Container for the API (`python:3.12-slim` + uvicorn) | Same code, same run, on any machine |
| `data/tickets.json` | 9 sample tickets (billing, technical, shipping, returns) | Realistic input data — each ticket now has an `"expected"` label; T-1003 names its real order ORD-1003 |
| `data/orders.json` | 8 real orders (ORD-1001–ORD-1008) with live statuses and a `"refund"` field on each order | The orders database the lookup answers from — and the sign-in view reads refunds from it |
| `knowledge/*.md` | 7 help articles (the RAG knowledge base) | Facts the responder is allowed to use |
| `memory.json` | Created on first run | Per-customer ticket history |
| `results.json` | Created by the API | Stored pipeline results, so `GET /tickets/{id}` can answer |
| `requirements.txt` | `openai`, `fastapi`, `uvicorn` | The runtime dependencies |
| `dev-requirements.txt` | `pytest`, `httpx` | Test-only dependencies |
| `.env.example` | Template for API key / base URL / model | Copy to `.env`; never commit a real key |

**Data flow, end to end:**

```
tickets.json ──┐
               ▼
main.py ──▶ SupportDesk.run(ticket)
               │  1. classifier.classify(llm, ticket) ──▶ LLM (JSON mode)
               │  2. retriever.search(ticket text) ──▶ knowledge/*.md (BM25)
               │  2b. order_lookup.find_order(order id) ──▶ data/orders.json
               │  3. responder.draft_reply(llm, ticket, docs, order) ──▶ LLM (grounded)
               ▼
          draft reply printed + ticket saved to memory.json
```

## 7. Demo it in 2 minutes

Say this, then run the commands:

> "This is a multi-agent support desk. A ticket comes in, three agents
> handle it in sequence, and the reply is grounded in our own help
> articles — the model can't invent policy."

```bash
# 1. Demo mode — no API key, fake LLM, four tickets (40 seconds)
python main.py --demo
# Watch the fourth ticket, T-1009 ("Do you accept Bitcoin?"): no help
#  article answers it, so the desk says "I don't know" and flags it
#  for a human instead of guessing. Run the demo a second time and
#  the replies open by recognising the returning customer.

# 2. One ticket, point at the steps on screen (30 seconds)
python main.py --ticket T-1004 --mock
# "Classifier says returns / high / negative. Retriever found the
#  damaged-item article. Responder drafted a reply citing it."

# 3. Show the memory (15 seconds)
cat memory.json
# "The same customer coming back tomorrow is recognised — C-101
#  already has two tickets on file — and the reply says so. Run the
#  same ticket a third time and the desk escalates it to a human:
#  a repeat unhappy customer gets a person, not another auto-reply."

# 4. Show how easy it is to extend (15 seconds)
ls knowledge/
# "Adding a help article is just dropping a markdown file here —
#  the retriever picks it up automatically."

# 5. Show the API (30 seconds) — in a second terminal:
uvicorn api:app --reload
curl -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" -d '{"ticket_id": "T-1001"}'
# "The same three agents, now behind an HTTP endpoint — any other
#  system can send a ticket and get the classification, the articles,
#  and the draft reply back as JSON."

# 6. Show the order lookup (30 seconds) — the moment that makes it real:
curl -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" -d '{"subject": "Where is my order #1001?", "message": "Hi, where is my order #1001?"}'
# "That is not a generic tracking article — the desk looked up order
#  ORD-1001 in the shop's own orders file and answered with its real
#  status: Shipped, with the items and the delivery date. Try #9999
#  and it says honestly that the order does not exist."

# 7. Show the sign-in on the site (30 seconds) — the unified view:
curl -X POST http://127.0.0.1:8000/login -H "Content-Type: application/json" -d '{"email": "sofia.khan@example.com", "password": "demo1234"}'
curl http://127.0.0.1:8000/customers/C-103/orders
# "One sign-in, and Sofia sees her delayed order, her refund
#  requested, and her past messages — order status, refund status,
#  and support history together, where most shops make you visit
#  three different pages."
```

If you have a real API key, run one ticket without `--mock` to show
the real LLM's much better reply.

## 8. Prompt engineering in this project

Prompt engineering is not "tricking the model" — it is writing clear
instructions. This project shows three techniques:

1. **Role + rules in the system prompt.** Each agent's system prompt
   starts with its job ("You are a support ticket classifier") and
   then lists exact rules. The responder's prompt is mostly rules:
   use only the articles, cite them, say "I don't know" otherwise.
2. **Structured output (JSON mode).** The classifier is asked for a
   JSON object with exactly three fields. That turns a free-text answer
   into data the program can use — no fragile parsing.
3. **Temperature 0.0.** All calls use temperature 0, which makes the
   model deterministic. For support answers you want the same correct
   answer every time, not creative variation.

## 9. What I would improve next (good to say to a manager)

- **Semantic search:** replace BM25 with embeddings so "money taken
  twice" matches "duplicate charge".
- **Auth and a real database:** the API has only a demo-only sign-in
  (three public accounts, one shared password printed on the page)
  and stores results in a JSON file — fine for a demo, but production
  needs real authentication and a database.

Already done (so don't list these as "next"): the API, the tests, the
eval script, structured logging, central config, CI, and Docker — see
§11. Also already done: the human-in-the-loop escalation (repeat
unhappy customers and unanswerable tickets go to a human), the
relevance threshold (a weak match no longer produces an answer from
the wrong article), and the real order lookup (tickets about an order
number are answered from the shop's own orders file, not a generic
article).

## 10. Likely manager questions — with short answers

**1. What is agentic AI, in one line?**
Software where an AI system plans steps, uses tools, and observes
results in a loop to reach a goal — instead of answering one prompt.

**2. Why three agents instead of one prompt?**
Separation of concerns: each agent has one job and one prompt, so each
is easy to test, fix, and explain. If classification breaks, only the
classifier changes.

**3. What is RAG and why did you use it?**
Retrieval-Augmented Generation: fetch facts from our own documents
first, then generate the answer from those facts. It stops the model
inventing policies or numbers.

**4. What is BM25?**
A classic keyword ranking algorithm. It scores documents by how often
query words appear, how rare those words are, and document length. Pure
Python, no libraries, ~60 lines.

**5. What happens when the knowledge base has no answer?**
The retriever has a relevance threshold (10.0, measured — see §12):
if the best article scores below it, nothing is returned, the
responder says it doesn't know, and the ticket is flagged for a
human. It never guesses, and it never answers from a barely-related
article. Ticket T-1009 proves it.

**6. How does it remember customers?**
After each ticket, the customer ID and ticket category are appended to
`memory.json`. Returning customers are recognised by ID — and the
memory is actually used: the reply acknowledges a repeat customer,
and a repeat unhappy customer with an urgent problem is escalated to
a human instead of getting another auto-reply.

**7. What does the mock LLM do?**
Rule-based fake used in demo mode — keyword matching for classification
and template replies for the responder. Lets the whole pipeline run with
no API key and no internet.

**8. How do you switch to a real model / Gemini?**
Environment variables only: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
Gemini works through its OpenAI-compatible endpoint — set the base URL
to `https://generativelanguage.googleapis.com/v1beta/openai/` and the
model to `gemini-2.0-flash`. No code changes.

**9. What are the limitations?**
Keyword search misses paraphrases; the mock LLM's answers are simple;
memory is a local JSON file, not a database; the API has only a demo
sign-in (three public accounts, one shared password on the page), not
real authentication; the escalation rules are fixed (high + negative +
2 prior tickets), not learned from data.

**10. How would you put this in production?**
Add a human approval queue for high-priority tickets, swap BM25 for
embeddings, store memory and results in a database, put auth on the API,
and keep API keys in a secret manager — never in code. The eval script
is already the measuring tool: run it after every change.

## 11. The service layer: API, tests, eval, CI, Docker

The agents are the interesting part, but a project only counts as
"real" when other people can run it, test it, and measure it. That is
what this layer does:

- **API (`api.py`).** FastAPI wraps the exact same `SupportDesk`
  pipeline — no duplicated logic. `POST /tickets` accepts a sample
  ticket id or an inline ticket, runs the pipeline quietly (the log
  lines still record every step), stores the result in `results.json`,
  and returns it as JSON. `GET /tickets/{id}` fetches a stored result.
  `GET /health` reports status, LLM mode, and how many tickets have
  been processed. The sign-in view is three more endpoints:
  `POST /login` checks one of three public demo accounts (password
  `demo1234`, printed on the homepage — demo-only, not real security),
  `GET /customers/{id}/orders` returns that customer's orders with
  their refund on each one, and `GET /customers/{id}/tickets` returns
  their processed tickets as answered or passed to a human. Without a
  key it runs in mock mode; with a key it uses the real LLM.
- **Config (`config.py`).** Every setting lives in one place and comes
  from the environment (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`,
  `MOCK_LLM`, `LOG_LEVEL`, `API_HOST`, `API_PORT`). Values are read
  through properties, so they are picked up whenever they are accessed
  — including after a `.env` file is loaded. No key is ever in code.
- **Logging.** Each pipeline step writes one structured log line, for
  example `step=classify ticket_id=T-1001 category=billing
  priority=high`. The CLI's THINK / ACT / OBSERVE prints are the
  human-readable demo; the log lines are the machine-readable record,
  and you can follow one ticket with `grep ticket_id=T-1001 app.log`.
- **Tests (`tests/`, 49 tests, pytest).** Six areas: the retriever
  ranks the expected article first (and stays silent below the
  relevance threshold); the classifier returns valid JSON with the
  three fields and the right categories; the end-to-end pipeline
  produces a reply that cites a real article; the order lookup
  extracts order ids, finds real orders, and honestly reports missing
  ones; the API endpoints answer, store, and 404 correctly, and
  identify customers from email or name (or treat a ticket as
  anonymous); the demo sign-in accepts the three demo accounts,
  rejects wrong credentials with a 401, returns only that customer's
  orders with their refund status, and lists their processed tickets
  (empty-safe); and the uniqueness behaviours —
  the desk says "I don't know" and escalates off-topic tickets,
  acknowledges repeat customers, and escalates repeat unhappy ones.
  A `conftest.py` fixture points memory and results at a temp folder,
  so tests never touch your real data. Run them with
  `python -m pytest -q` (use the project's `venv` Python).
- **Eval (`eval.py`).** Runs all 9 tickets in mock mode and prints a
  table: expected vs predicted category, top-1 article, hit or miss,
  and whether the ticket was escalated — then the four headline
  numbers (classification accuracy 9/9, retrieval top-1 hit rate 9/9,
  honesty 1/1 in mock mode, plus order lookup 2/2 from two extra
  probes: one order that exists, one that does not). Each ticket
  carries its own `"expected"` label in `data/tickets.json`, so the
  measurement is data-driven. If you change the classifier or the
  retriever, run the eval and watch the numbers move — that is the
  honest way to show an improvement.
- **CI (`.github/workflows/ci.yml`).** On every push, GitHub Actions
  installs the dependencies, runs pytest, and runs the demo. It uses
  mock mode, so CI needs no API key and no secrets.
- **Docker (`Dockerfile`).** `python:3.12-slim`, installs
  `requirements.txt`, runs `uvicorn api:app`. One command builds it,
  one runs it, on any machine: `docker build -t support-desk .` then
  `docker run -p 8000:8000 support-desk`. Pass a real key with
  `-e LLM_API_KEY=...` when you want the real LLM.

How to explain this layer to your manager, in one breath:

> "The agents are the core, but I also made it a real service: an API
> so other systems can send tickets, 49 automated tests, an eval script
> that scores the pipeline 9 out of 9 on classification and retrieval
> plus 1 out of 1 on honesty and 2 out of 2 on order lookup, CI that
> checks every push, and a Docker image so it runs the same everywhere.
> The eval numbers are in mock mode — the next step is running the same
> eval against the real model."

**11. Why have both a CLI and an API?**
The CLI is the demo and the debugging tool — you see every step. The
API is how a real product would use the desk: another system sends a
ticket and gets structured JSON back. Same pipeline, two doors in.

**12. How do you know it works — really?**
Three ways: the tests (49 passing) pin the behaviour, the eval
measures it (9/9 classification, 9/9 retrieval, honesty 1/1, order
lookup 2/2 in mock mode), and CI re-runs both on every push. If a
change breaks something, one of the three catches it before a human
has to.

**13. What does "mock mode" mean for these numbers?**
The eval and the tests run against the rule-based mock LLM, not a real
model — so the 100% scores prove the pipeline, the retrieval, and the
grounding work, not that a real model classifies perfectly. The honest
next step is running the same eval with a real API key and reporting
that number too.

**14. What makes this project different from other support-bot demos?**
Five things, all measured: it refuses to answer when it has no
answer (the relevance threshold + "I don't know" + escalation,
measured by the honesty score); it uses the customer's history
instead of just storing it (the reply acknowledges repeat customers,
and repeat unhappy customers are escalated); it answers "where is my
order?" from a real orders file instead of a generic article (and
honestly says when the order does not exist, measured by the order
lookup score); one sign-in shows order status, refund status, and
past messages together, where most shops split them across three
different pages — and the sign-in needs no signup, the demo accounts
are printed on the page; and everything is proven with numbers (49
tests, 9/9 + 9/9 + 1/1 + 2/2 eval, CI on every push). Most demos only
show the happy path — this one is built around the hard cases.

**15. How did you pick the relevance threshold of 10.0?**
By measuring, not guessing. I scored all 8 real tickets (15.49–
36.45) and several off-topic questions (at most 9.18 for the Bitcoin
ticket, 2.40 for the short probes) and put the threshold between
them. The number is written in `agents/retriever.py` with the
measurements, so anyone can re-check it — and it must be
re-measured if the knowledge base changes.

**16. Exactly when does a ticket go to a human?**
Two rules, both visible in the result as `needs_human_review` with
a reason: (1) no help article scores above the relevance threshold
AND the order lookup did not answer the ticket either — the desk said
"I don't know"; (2) the customer is unhappy (negative sentiment)
about an urgent problem (high priority) and has already written twice
before. Everything else gets a draft reply the human can send as-is.
Note the first rule carefully: a ticket about a real order that the
lookup found is answered, so it is not escalated — flagging it as "no
article answers this" would be a reason that is not true.

**17. How does the order lookup work?**
The orchestrator extracts an order number from the ticket text with a
small regex — it matches `ORD-1001` and `#1001`, never a bare number
("I have 2 orders" is not an order id) — then looks it up in
`data/orders.json`, a file of 8 real orders with items, totals, live
statuses, carriers, and tracking numbers. The order (or the fact that
it does not exist) goes into the responder's prompt as an ORDER
DETAILS block, so the reply states the order's real status, items,
and delivery date. The result also carries the order as `"order"` and
the mentioned id as `"order_id_mentioned"`, so the homepage can show
an "Order found" panel.

**18. What happens when the order number doesn't exist?**
The lookup returns nothing, the prompt says so plainly, and the reply
says "I couldn't find order ORD-9999" and asks the customer to check
the number. It never invents a status, tracking number, or delivery
date — and the ticket is escalated to a human, because no article and
no order answered it. The eval measures this: the order-lookup score
is 2/2 (one found, one not found).

**19. How does the site know who I am now?**
The web form has optional Name and Email fields. An email (lowercased)
becomes the customer id, so the same person writing twice is
recognised as a repeat customer; a name alone becomes a `C-first-last`
id. With neither, the ticket is anonymous: no history is recorded and
the reply never claims a repeat visit that did not happen. Before
this fix every form ticket shared one default customer id, so a
first-time stranger's reply wrongly said "you've written to us 1
time(s) before".

**20. How does a customer check an order or a refund?**
They sign in on the homepage — no sign-up, the demo accounts are
printed on the card — and one page shows their orders with status
and delivery date, each order's refund status, and their past
messages with us.
