"""
agents/classifier.py — Agent 1: understand the ticket.

Job: read a support ticket and answer three questions about it:
  * category  — billing / technical / shipping / returns / general
  * priority  — high / normal / low  (how urgent is it?)
  * sentiment — positive / neutral / negative  (how does the customer feel?)

How it works: it sends the ticket to the LLM with a prompt that asks for
a JSON object with exactly those three fields. Using JSON mode means the
answer comes back in a fixed shape the code can parse reliably — no
guessing, no free-text parsing.

This is the "THINK" step of the agent loop: the agent looks at the input
and decides what kind of problem it is before doing anything else.
"""

import json

# The system prompt is the agent's job description. It tells the LLM what
# its role is and exactly what shape the answer must take.
SYSTEM_PROMPT = """You are a support ticket classifier. ROLE: classifier

Read the customer ticket and classify it. Reply with a JSON object only,
with exactly these three fields:
{
  "category": "billing" | "technical" | "shipping" | "returns" | "general",
  "priority": "high" | "normal" | "low",
  "sentiment": "positive" | "neutral" | "negative"
}

Rules:
- "billing" is about charges, refunds, invoices, subscriptions, payments.
- "technical" is about logins, passwords, errors, crashes, the app or website.
- "shipping" is about delivery, tracking, addresses, lost packages.
- "returns" is about returning, exchanging, or damaged items.
- Priority is "high" when the customer is blocked, charged wrongly, or upset;
  "low" when it is a simple question; otherwise "normal".
- Sentiment is "negative" when the customer sounds angry or frustrated,
  "positive" when they sound happy or thankful, otherwise "neutral".
"""


def classify(llm, ticket):
    """Classify one ticket. Returns a dict: category, priority, sentiment."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"TICKET:\n{ticket['subject']}\n{ticket['message']}",
        },
    ]

    # temperature=0.0 makes the answer deterministic; json_mode=True makes
    # the API return a JSON object instead of free text.
    raw = llm.chat(messages, temperature=0.0, json_mode=True)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Safety net: if the model ever returns broken JSON, fall back to
        # a neutral classification instead of crashing the pipeline.
        result = {"category": "general", "priority": "normal", "sentiment": "neutral"}

    # Only keep the three fields we asked for, with safe defaults.
    return {
        "category": result.get("category", "general"),
        "priority": result.get("priority", "normal"),
        "sentiment": result.get("sentiment", "neutral"),
    }
